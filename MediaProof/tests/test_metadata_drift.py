"""
Tests for Structured Technical Metadata Extraction via FFprobe (Phase 1.4).

Covers all 12 required scenarios:
1. Valid container, video, and audio metadata.
2. Multiple video and audio streams.
3. Missing optional fields.
4. Missing FFprobe executable.
5. Timeout.
6. Nonzero FFprobe exit status.
7. Malformed JSON.
8. Unexpected JSON structure.
9. Invalid or inaccessible input.
10. Correct parsing of numeric values and frame-rate fractions.
11. Preservation of stream indices and order.
12. Ensuring failures remain unavailable/error states.
Plus real media fixture execution when FFprobe is present.
"""

import json
import math
from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from forensics.metadata_drift import (
    AudioStreamMetadata,
    ContainerMetadata,
    STATUS_FFPROBE_UNAVAILABLE,
    STATUS_INVALID_INPUT,
    STATUS_MALFORMED_OUTPUT,
    STATUS_PROCESS_FAILURE,
    STATUS_SUCCESS,
    STATUS_SUCCESS_WITH_MISSING_FIELDS,
    STATUS_TIMEOUT,
    TechnicalMetadataResult,
    VideoStreamMetadata,
    extract_encoder_tag,
    extract_technical_metadata,
    parse_fraction_or_float,
    parse_optional_float,
    parse_optional_int,
)
from media.ffmpeg import FFmpegTimeoutError


@pytest.fixture
def dummy_media_file(tmp_path: Path) -> Path:
    """Creates a dummy media file on disk for file-existence validation."""
    media_file = tmp_path / "sample_video.mp4"
    media_file.write_bytes(b"\x00" * 1024)
    return media_file


@pytest.fixture
def synthetic_video_file(tmp_path: Path) -> Path:
    """Creates a real synthetic playable video file with OpenCV."""
    video_path = tmp_path / "synthetic_fixture.mp4"
    fps = 10.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (64, 64))
    for i in range(15):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, : (i * 4)] = 200
        out.write(frame)
    out.release()
    return video_path


# ==============================================================================
# 1. Valid container, video, and audio metadata
# ==============================================================================

def test_valid_container_video_audio_metadata(dummy_media_file: Path):
    mock_ffprobe_dict = {
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "10.500000",
            "bit_rate": "1250000",
            "start_time": "0.000000",
            "size": "1640625",
            "tags": {
                "major_brand": "isom",
                "encoder": "Lavf60.3.100",
            },
        },
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "profile": "High",
                "level": 40,
                "pix_fmt": "yuv420p",
                "width": 1920,
                "height": 1080,
                "sample_aspect_ratio": "1:1",
                "display_aspect_ratio": "16:9",
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "30/1",
                "color_range": "tv",
                "color_space": "bt709",
                "color_transfer": "bt709",
                "color_primaries": "bt709",
                "nb_frames": "315",
                "time_base": "1/30000",
                "duration": "10.500000",
                "bit_rate": "1120000",
                "tags": {"handler_name": "VideoHandler"},
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
                "channel_layout": "stereo",
                "duration": "10.510000",
                "bit_rate": "128000",
                "tags": {"language": "eng"},
            },
        ],
    }

    completed_proc = subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=0,
        stdout=json.dumps(mock_ffprobe_dict),
        stderr="",
    )

    with patch("forensics.metadata_drift.run_ffmpeg_command", return_value=completed_proc), \
         patch("forensics.metadata_drift.get_ffmpeg_version", return_value="ffprobe 7.0.2"):
        res = extract_technical_metadata(dummy_media_file, include_raw=True)

    assert res.available is True
    assert res.status == STATUS_SUCCESS
    assert res.file_path == str(dummy_media_file.resolve())
    assert res.ffprobe_version == "ffprobe 7.0.2"
    assert res.video_stream_count == 1
    assert res.audio_stream_count == 1
    assert res.other_stream_count == 0

    # Container assertions
    assert res.container is not None
    assert res.container.format_name == "mov,mp4,m4a,3gp,3g2,mj2"
    assert res.container.duration_seconds == 10.5
    assert res.container.bitrate == 1250000
    assert res.container.start_time_seconds == 0.0
    assert res.container.size_bytes == 1640625
    assert res.container.encoder == "Lavf60.3.100"

    # Video stream assertions
    v = res.video_streams[0]
    assert v.stream_index == 0
    assert v.codec_name == "h264"
    assert v.profile == "High"
    assert v.level == 40
    assert v.pixel_format == "yuv420p"
    assert v.width == 1920
    assert v.height == 1080
    assert v.sample_aspect_ratio == "1:1"
    assert v.display_aspect_ratio == "16:9"
    assert abs(v.avg_frame_rate - (30000 / 1001)) < 1e-5
    assert v.r_frame_rate == 30.0
    assert v.color_range == "tv"
    assert v.color_space == "bt709"
    assert v.frame_count == 315
    assert v.time_base == "1/30000"
    assert v.duration_seconds == 10.5
    assert v.bitrate == 1120000

    # Audio stream assertions
    a = res.audio_streams[0]
    assert a.stream_index == 1
    assert a.codec_name == "aac"
    assert a.sample_rate == 48000
    assert a.channels == 2
    assert a.channel_layout == "stereo"
    assert a.duration_seconds == 10.51
    assert a.bitrate == 128000

    # Verify serialization
    res_dict = res.to_dict()
    assert res_dict["available"] is True
    assert res_dict["status"] == STATUS_SUCCESS
    assert res_dict["container"]["format_name"] == "mov,mp4,m4a,3gp,3g2,mj2"
    assert res_dict["raw_metadata"] == mock_ffprobe_dict


# ==============================================================================
# 2. Multiple video and audio streams
# ==============================================================================

def test_multiple_video_and_audio_streams(dummy_media_file: Path):
    mock_dict = {
        "format": {"format_name": "matroska,webm"},
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": 48000,
                "channels": 2,
            },
            {
                "index": 2,
                "codec_type": "video",
                "codec_name": "hevc",
                "width": 3840,
                "height": 2160,
            },
            {
                "index": 3,
                "codec_type": "audio",
                "codec_name": "ac3",
                "sample_rate": 44100,
                "channels": 6,
            },
            {
                "index": 4,
                "codec_type": "subtitle",
                "codec_name": "subrip",
            },
        ],
    }

    proc = subprocess.CompletedProcess(args=["ffprobe"], returncode=0, stdout=json.dumps(mock_dict), stderr="")
    with patch("forensics.metadata_drift.run_ffmpeg_command", return_value=proc):
        res = extract_technical_metadata(dummy_media_file)

    assert res.available is True
    assert res.video_stream_count == 2
    assert res.audio_stream_count == 2
    assert res.other_stream_count == 1

    # Verify stream ordering and index preservation
    assert [v.stream_index for v in res.video_streams] == [0, 2]
    assert [v.codec_name for v in res.video_streams] == ["h264", "hevc"]
    assert [v.width for v in res.video_streams] == [1920, 3840]

    assert [a.stream_index for a in res.audio_streams] == [1, 3]
    assert [a.codec_name for a in res.audio_streams] == ["aac", "ac3"]
    assert [a.channels for a in res.audio_streams] == [2, 6]


# ==============================================================================
# 3. Missing optional fields (No Fabricated Defaults)
# ==============================================================================

def test_missing_optional_fields_preserves_none(dummy_media_file: Path):
    mock_dict = {
        "format": {
            "format_name": "mp4",
            # duration, bit_rate, start_time, tags, size are all missing
        },
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                # avg_frame_rate, duration, bit_rate, color properties missing
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": 44100,
                "channels": 2,
                # channel_layout, duration, bit_rate missing
            },
        ],
    }

    proc = subprocess.CompletedProcess(args=["ffprobe"], returncode=0, stdout=json.dumps(mock_dict), stderr="")
    with patch("forensics.metadata_drift.run_ffmpeg_command", return_value=proc):
        res = extract_technical_metadata(dummy_media_file)

    assert res.available is True
    # Container duration and bit_rate must remain None, not 0.0 or 0
    assert res.container.duration_seconds is None
    assert res.container.bitrate is None
    assert res.container.start_time_seconds is None
    assert res.container.encoder is None

    # Video missing fields must remain None
    v = res.video_streams[0]
    assert v.duration_seconds is None
    assert v.bitrate is None
    assert v.avg_frame_rate is None
    assert v.color_space is None
    assert v.frame_count is None

    # Audio missing fields must remain None
    a = res.audio_streams[0]
    assert a.channel_layout is None
    assert a.duration_seconds is None
    assert a.bitrate is None


def test_missing_essential_field_marks_status():
    """Missing essential container or stream field produces STATUS_SUCCESS_WITH_MISSING_FIELDS."""
    mock_dict = {
        "format": {},  # Missing format_name!
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                # Missing codec_name, width, height!
            }
        ],
    }
    dummy_file = Path("any_file.mp4")
    proc = subprocess.CompletedProcess(args=["ffprobe"], returncode=0, stdout=json.dumps(mock_dict), stderr="")
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.is_file", return_value=True), \
         patch("forensics.metadata_drift.run_ffmpeg_command", return_value=proc):
        res = extract_technical_metadata(dummy_file)

    assert res.available is True
    assert res.status == STATUS_SUCCESS_WITH_MISSING_FIELDS
    assert res.container.format_name is None


# ==============================================================================
# 4. Missing FFprobe executable
# ==============================================================================

def test_missing_ffprobe_executable(dummy_media_file: Path):
    with patch("forensics.metadata_drift.get_ffprobe_path", return_value=None):
        res = extract_technical_metadata(dummy_media_file)

    assert res.available is False
    assert res.status == STATUS_FFPROBE_UNAVAILABLE
    assert res.container is None
    assert res.video_streams == []
    assert res.audio_streams == []
    assert "not found" in res.details["error"]


def test_specified_missing_ffprobe_binary(dummy_media_file: Path):
    res = extract_technical_metadata(dummy_media_file, ffprobe_binary="Z:/non_existent_ffprobe.exe")
    assert res.available is False
    assert res.status == STATUS_FFPROBE_UNAVAILABLE
    assert "not found" in res.details["error"]


# ==============================================================================
# 5. Timeout
# ==============================================================================

def test_ffprobe_timeout(dummy_media_file: Path):
    with patch("forensics.metadata_drift.run_ffmpeg_command", side_effect=FFmpegTimeoutError("Subprocess timed out")):
        res = extract_technical_metadata(dummy_media_file, timeout=5.0)

    assert res.available is False
    assert res.status == STATUS_TIMEOUT
    assert res.container is None
    assert res.video_streams == []
    assert res.audio_streams == []
    assert "timed out after 5.0 seconds" in res.details["error"]
    assert res.details["timeout_seconds"] == 5.0


# ==============================================================================
# 6. Nonzero FFprobe exit status
# ==============================================================================

def test_nonzero_ffprobe_exit_status(dummy_media_file: Path):
    failed_proc = subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=1,
        stdout="",
        stderr="sample_video.mp4: Invalid data found when processing input",
    )

    with patch("forensics.metadata_drift.run_ffmpeg_command", return_value=failed_proc):
        res = extract_technical_metadata(dummy_media_file)

    assert res.available is False
    assert res.status == STATUS_PROCESS_FAILURE
    assert res.container is None
    assert res.video_streams == []
    assert res.details["returncode"] == 1
    assert "Invalid data found" in res.details["stderr"]


# ==============================================================================
# 7. Malformed JSON
# ==============================================================================

def test_malformed_json_output(dummy_media_file: Path):
    broken_proc = subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=0,
        stdout="{format: {incomplete json",
        stderr="",
    )

    with patch("forensics.metadata_drift.run_ffmpeg_command", return_value=broken_proc):
        res = extract_technical_metadata(dummy_media_file)

    assert res.available is False
    assert res.status == STATUS_MALFORMED_OUTPUT
    assert res.container is None
    assert "Malformed JSON" in res.details["error"]


# ==============================================================================
# 8. Unexpected JSON structure
# ==============================================================================

@pytest.mark.parametrize(
    "unexpected_payload",
    [
        "[]",  # Top level list instead of dict
        json.dumps("some string"),
        json.dumps({"format": "not_a_dict", "streams": []}),
        json.dumps({"format": {}, "streams": "not_a_list"}),
    ],
)
def test_unexpected_json_structure(dummy_media_file: Path, unexpected_payload: str):
    proc = subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=0,
        stdout=unexpected_payload,
        stderr="",
    )

    with patch("forensics.metadata_drift.run_ffmpeg_command", return_value=proc):
        res = extract_technical_metadata(dummy_media_file)

    assert res.available is False
    assert res.status == STATUS_MALFORMED_OUTPUT
    assert res.container is None


# ==============================================================================
# 9. Invalid or inaccessible input
# ==============================================================================

def test_invalid_input_path_scenarios(tmp_path: Path):
    # Empty string
    res_empty = extract_technical_metadata("")
    assert res_empty.available is False
    assert res_empty.status == STATUS_INVALID_INPUT

    # None path
    res_none = extract_technical_metadata(None)  # type: ignore
    assert res_none.available is False
    assert res_none.status == STATUS_INVALID_INPUT

    # Non-existent file
    res_missing = extract_technical_metadata(tmp_path / "does_not_exist_file.mp4")
    assert res_missing.available is False
    assert res_missing.status == STATUS_INVALID_INPUT
    assert "does not exist" in res_missing.details["error"]

    # Directory instead of file
    res_dir = extract_technical_metadata(tmp_path)
    assert res_dir.available is False
    assert res_dir.status == STATUS_INVALID_INPUT
    assert "not a regular file" in res_dir.details["error"]


# ==============================================================================
# 10. Correct parsing of numeric values and frame-rate fractions
# ==============================================================================

def test_parse_fraction_or_float():
    # Valid fractions
    assert abs(parse_fraction_or_float("30000/1001") - 29.97002997) < 1e-5
    assert parse_fraction_or_float("25/1") == 25.0
    assert parse_fraction_or_float("60/2") == 30.0
    assert parse_fraction_or_float("24") == 24.0
    assert parse_fraction_or_float(29.97) == 29.97
    assert parse_fraction_or_float(30) == 30.0

    # Invalid fractions & sentinel values must return None (no fabricated defaults)
    assert parse_fraction_or_float("0/0") is None
    assert parse_fraction_or_float("N/A") is None
    assert parse_fraction_or_float("unknown") is None
    assert parse_fraction_or_float("none") is None
    assert parse_fraction_or_float("") is None
    assert parse_fraction_or_float(None) is None
    assert parse_fraction_or_float("100/0") is None  # Division by zero
    assert parse_fraction_or_float("invalid/fraction") is None
    assert parse_fraction_or_float(float("nan")) is None
    assert parse_fraction_or_float(float("inf")) is None
    assert parse_fraction_or_float(True) is None  # Bool guards


def test_parse_optional_float():
    assert parse_optional_float("12.345") == 12.345
    assert parse_optional_float("0") == 0.0
    assert parse_optional_float(0.0) == 0.0
    assert parse_optional_float("0.000000") == 0.0

    # Sentinels
    assert parse_optional_float("N/A") is None
    assert parse_optional_float("unknown") is None
    assert parse_optional_float(None) is None
    assert parse_optional_float(float("nan")) is None
    assert parse_optional_float(float("-inf")) is None
    assert parse_optional_float(False) is None


def test_parse_optional_int():
    assert parse_optional_int("128000") == 128000
    assert parse_optional_int("128000.0") == 128000
    assert parse_optional_int(128000) == 128000
    assert parse_optional_int("0") == 0

    assert parse_optional_int("N/A") is None
    assert parse_optional_int("unknown") is None
    assert parse_optional_int(None) is None
    assert parse_optional_int(True) is None


def test_extract_encoder_tag():
    assert extract_encoder_tag({"encoder": "Lavf60.3.100"}) == "Lavf60.3.100"
    assert extract_encoder_tag({"ENCODER": "HandBrake 1.5.1"}) == "HandBrake 1.5.1"
    assert extract_encoder_tag({"encoded_by": "Adobe Premiere"}) == "Adobe Premiere"
    assert extract_encoder_tag({"handler_name": "VideoHandler"}) == "VideoHandler"
    assert extract_encoder_tag({}) is None
    assert extract_encoder_tag(None) is None


# ==============================================================================
# 11. Preservation of stream indices and order
# ==============================================================================

def test_stream_indices_and_order_preserved(dummy_media_file: Path):
    # Streams intentionally provided out-of-order in the JSON payload
    mock_dict = {
        "format": {"format_name": "mp4"},
        "streams": [
            {"index": 3, "codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720},
            {"index": 1, "codec_type": "audio", "codec_name": "aac", "sample_rate": 44100, "channels": 2},
            {"index": 0, "codec_type": "video", "codec_name": "vp9", "width": 1920, "height": 1080},
            {"index": 2, "codec_type": "audio", "codec_name": "opus", "sample_rate": 48000, "channels": 2},
        ],
    }

    proc = subprocess.CompletedProcess(args=["ffprobe"], returncode=0, stdout=json.dumps(mock_dict), stderr="")
    with patch("forensics.metadata_drift.run_ffmpeg_command", return_value=proc):
        res = extract_technical_metadata(dummy_media_file)

    assert res.available is True
    # Video streams should be sorted chronologically by index: index 0 (vp9) first, then index 3 (h264)
    assert len(res.video_streams) == 2
    assert res.video_streams[0].stream_index == 0
    assert res.video_streams[0].codec_name == "vp9"
    assert res.video_streams[1].stream_index == 3
    assert res.video_streams[1].codec_name == "h264"

    # Audio streams should be sorted chronologically by index: index 1 (aac), then index 2 (opus)
    assert len(res.audio_streams) == 2
    assert res.audio_streams[0].stream_index == 1
    assert res.audio_streams[0].codec_name == "aac"
    assert res.audio_streams[1].stream_index == 2
    assert res.audio_streams[1].codec_name == "opus"


# ==============================================================================
# 12. Ensuring failures remain unavailable/error states (never false success)
# ==============================================================================

@pytest.mark.parametrize(
    "error_scenario,expected_status",
    [
        ("invalid_path", STATUS_INVALID_INPUT),
        ("missing_ffprobe", STATUS_FFPROBE_UNAVAILABLE),
        ("timeout", STATUS_TIMEOUT),
        ("process_failure", STATUS_PROCESS_FAILURE),
        ("malformed_json", STATUS_MALFORMED_OUTPUT),
    ],
)
def test_failures_remain_unavailable(dummy_media_file: Path, error_scenario: str, expected_status: str):
    if error_scenario == "invalid_path":
        res = extract_technical_metadata("path/to/nonexistent/media.mp4")
    elif error_scenario == "missing_ffprobe":
        with patch("forensics.metadata_drift.get_ffprobe_path", return_value=None):
            res = extract_technical_metadata(dummy_media_file)
    elif error_scenario == "timeout":
        with patch("forensics.metadata_drift.run_ffmpeg_command", side_effect=FFmpegTimeoutError("timeout")):
            res = extract_technical_metadata(dummy_media_file)
    elif error_scenario == "process_failure":
        proc = subprocess.CompletedProcess(args=["ffprobe"], returncode=2, stdout="", stderr="Corrupted stream")
        with patch("forensics.metadata_drift.run_ffmpeg_command", return_value=proc):
            res = extract_technical_metadata(dummy_media_file)
    elif error_scenario == "malformed_json":
        proc = subprocess.CompletedProcess(args=["ffprobe"], returncode=0, stdout="not valid json", stderr="")
        with patch("forensics.metadata_drift.run_ffmpeg_command", return_value=proc):
            res = extract_technical_metadata(dummy_media_file)

    # CRITICAL: A failure must NEVER be reported as available=True or have fabricated empty metadata
    assert res.available is False
    assert res.status == expected_status
    assert res.container is None
    assert res.video_streams == []
    assert res.audio_streams == []
    assert "error" in res.details


# ==============================================================================
# Integration: Real Media Fixture with Native FFprobe
# ==============================================================================

def test_real_media_fixture_extraction(synthetic_video_file: Path):
    from media.ffmpeg import get_ffprobe_path
    ffprobe_path = get_ffprobe_path()
    if not ffprobe_path or not ffprobe_path.exists():
        pytest.skip("Local FFprobe binary not available for end-to-end fixture test")

    res = extract_technical_metadata(synthetic_video_file, include_raw=True)

    assert res.available is True
    assert res.status in (STATUS_SUCCESS, STATUS_SUCCESS_WITH_MISSING_FIELDS)
    assert res.file_path == str(synthetic_video_file.resolve())
    assert res.video_stream_count >= 1
    assert res.container is not None
    assert res.container.format_name is not None

    v = res.video_streams[0]
    assert v.width == 64
    assert v.height == 64
    assert v.stream_index == 0
    assert v.avg_frame_rate is not None
    assert v.avg_frame_rate > 0.0


# ==============================================================================
# Phase 1.4 Corrective Step Regression Tests
# ==============================================================================

def test_version_detection_exception_allows_successful_extraction(dummy_media_file: Path):
    """1. Version detection raises an exception, but metadata extraction subsequently succeeds."""
    mock_payload = {
        "format": {"format_name": "mp4", "duration": "5.0"},
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720}
        ],
    }
    proc = subprocess.CompletedProcess(args=["ffprobe"], returncode=0, stdout=json.dumps(mock_payload), stderr="")

    with patch("forensics.metadata_drift.get_ffmpeg_version", side_effect=RuntimeError("Subprocess failed")), \
         patch("forensics.metadata_drift.run_ffmpeg_command", return_value=proc):
        res = extract_technical_metadata(dummy_media_file)

    assert res.available is True
    assert res.status == STATUS_SUCCESS
    # Version must NOT be fabricated; should remain None
    assert res.ffprobe_version is None
    assert "version_diagnostic" in res.details
    assert "Version detection failed" in res.details["version_diagnostic"]
    assert res.video_stream_count == 1
    assert res.video_streams[0].codec_name == "h264"


def test_version_detection_fails_and_ffprobe_invocation_fails(dummy_media_file: Path):
    """2. Version detection fails and the actual FFprobe invocation also fails."""
    failed_proc = subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=1,
        stdout="",
        stderr="Input format not supported",
    )

    with patch("forensics.metadata_drift.get_ffmpeg_version", return_value="Unknown"), \
         patch("forensics.metadata_drift.run_ffmpeg_command", return_value=failed_proc):
        res = extract_technical_metadata(dummy_media_file)

    assert res.available is False
    assert res.status == STATUS_PROCESS_FAILURE
    assert res.ffprobe_version is None
    assert res.container is None
    assert res.video_streams == []
    assert res.audio_streams == []
    assert "version_diagnostic" in res.details
    assert res.details["returncode"] == 1


def test_empty_json_object_rejected(dummy_media_file: Path):
    """3. Empty JSON object {} is rejected as MALFORMED_OUTPUT (never available=True)."""
    proc = subprocess.CompletedProcess(args=["ffprobe"], returncode=0, stdout="{}", stderr="")

    with patch("forensics.metadata_drift.run_ffmpeg_command", return_value=proc):
        res = extract_technical_metadata(dummy_media_file)

    assert res.available is False
    assert res.status == STATUS_MALFORMED_OUTPUT
    assert res.container is None
    assert res.video_streams == []
    assert res.audio_streams == []
    assert "empty JSON object" in res.details["error"]


def test_missing_both_usable_format_and_streams_rejected(dummy_media_file: Path):
    """4. Missing both usable format information and usable stream information."""
    # Format has no usable keys, streams is empty or has unusable items
    payloads = [
        {"format": {}, "streams": []},
        {"format": {}, "streams": [{}]},
        {"format": {"tags": {}}, "streams": []},
    ]

    for p in payloads:
        proc = subprocess.CompletedProcess(args=["ffprobe"], returncode=0, stdout=json.dumps(p), stderr="")
        with patch("forensics.metadata_drift.run_ffmpeg_command", return_value=proc):
            res = extract_technical_metadata(dummy_media_file)

        assert res.available is False, f"Failed for payload: {p}"
        assert res.status == STATUS_MALFORMED_OUTPUT
        assert res.container is None
        assert res.video_streams == []
        assert res.audio_streams == []
        assert "usable container-format" in res.details["error"]


def test_valid_format_only_output(dummy_media_file: Path):
    """5. Valid format-only output remains usable with SUCCESS_WITH_MISSING_FIELDS."""
    payload = {
        "format": {
            "format_name": "mp3",
            "duration": "180.25",
            "bit_rate": "320000",
            "size": "7210000",
        },
        "streams": [],
    }
    proc = subprocess.CompletedProcess(args=["ffprobe"], returncode=0, stdout=json.dumps(payload), stderr="")

    with patch("forensics.metadata_drift.run_ffmpeg_command", return_value=proc):
        res = extract_technical_metadata(dummy_media_file)

    assert res.available is True
    assert res.status == STATUS_SUCCESS_WITH_MISSING_FIELDS
    assert res.container is not None
    assert res.container.format_name == "mp3"
    assert res.container.duration_seconds == 180.25
    assert res.container.bitrate == 320000
    assert res.video_stream_count == 0
    assert res.audio_stream_count == 0
    assert res.video_streams == []
    assert res.audio_streams == []


def test_valid_stream_only_output(dummy_media_file: Path):
    """6. Valid stream-only output remains usable with SUCCESS_WITH_MISSING_FIELDS."""
    payload = {
        "format": {},  # Container format absent or empty
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "24/1",
            }
        ],
    }
    proc = subprocess.CompletedProcess(args=["ffprobe"], returncode=0, stdout=json.dumps(payload), stderr="")

    with patch("forensics.metadata_drift.run_ffmpeg_command", return_value=proc):
        res = extract_technical_metadata(dummy_media_file)

    assert res.available is True
    assert res.status == STATUS_SUCCESS_WITH_MISSING_FIELDS
    assert res.container is not None
    assert res.container.format_name is None
    assert res.video_stream_count == 1
    assert res.video_streams[0].codec_name == "h264"
    assert res.video_streams[0].width == 1920
    assert res.video_streams[0].height == 1080
    assert res.video_streams[0].avg_frame_rate == 24.0


def test_existing_successful_extraction_missing_optional_fields(dummy_media_file: Path):
    """7. Existing successful extraction with missing optional fields."""
    payload = {
        "format": {
            "format_name": "mp4",
            # duration, bit_rate, tags are omitted optional fields
        },
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                # profile, level, color properties are omitted optional fields
            }
        ],
    }
    proc = subprocess.CompletedProcess(args=["ffprobe"], returncode=0, stdout=json.dumps(payload), stderr="")

    with patch("forensics.metadata_drift.run_ffmpeg_command", return_value=proc):
        res = extract_technical_metadata(dummy_media_file)

    assert res.available is True
    assert res.status == STATUS_SUCCESS
    assert res.container.format_name == "mp4"
    assert res.container.duration_seconds is None
    assert res.video_streams[0].codec_name == "h264"
    assert res.video_streams[0].profile is None
    assert res.video_streams[0].color_space is None


@pytest.mark.parametrize(
    "malformed_payload",
    [
        "[]",
        json.dumps("a raw string"),
        json.dumps(12345),
        json.dumps({"format": "not_a_dict"}),
        json.dumps({"streams": "not_a_list"}),
        json.dumps({"format": {"format_name": "mp4"}, "streams": ["not_a_dict_stream"]}),
        "{}",
        json.dumps({"format": {}, "streams": []}),
    ],
)
def test_all_malformed_output_cases_return_unavailable(dummy_media_file: Path, malformed_payload: str):
    """8. All malformed-output cases return available=False and MALFORMED_OUTPUT."""
    proc = subprocess.CompletedProcess(args=["ffprobe"], returncode=0, stdout=malformed_payload, stderr="")

    with patch("forensics.metadata_drift.run_ffmpeg_command", return_value=proc):
        res = extract_technical_metadata(dummy_media_file)

    assert res.available is False
    assert res.status == STATUS_MALFORMED_OUTPUT
    assert res.container is None
    assert res.video_streams == []
    assert res.audio_streams == []
    assert "error" in res.details

