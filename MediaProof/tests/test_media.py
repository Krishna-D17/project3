import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app
from media.ffmpeg import get_ffmpeg_path, get_ffprobe_path, get_ffmpeg_status
from media.service import MediaService

client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_MEDIA_DIR = PROJECT_ROOT / "data" / "temp" / "test_media"


def is_ffmpeg_available() -> bool:
    status = get_ffmpeg_status()
    return bool(status["ffmpeg_available"] and status["ffprobe_available"])


def test_ffmpeg_executable_discovery():
    status = get_ffmpeg_status()
    assert "ffmpeg_available" in status
    assert "ffprobe_available" in status
    assert "ffmpeg_path" in status
    assert "ffprobe_path" in status


def test_nonexistent_media_path():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")
    fake_path = PROJECT_ROOT / "data" / "temp" / "nonexistent_file_xyz123.mp4"
    with pytest.raises(FileNotFoundError):
        MediaService.probe_file(fake_path)


def test_probe_valid_wav_file():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")
    wav_file = TEST_MEDIA_DIR / "test_audio.wav"
    if not wav_file.exists():
        pytest.skip("Test WAV file not generated")

    result = MediaService.probe_file(wav_file)
    assert result.filename == "test_audio.wav"
    assert result.has_audio is True
    assert result.has_video is False
    assert result.is_supported is True
    assert len(result.audio_streams) >= 1
    assert result.audio_streams[0].sample_rate == 44100


def test_probe_valid_mp4_file():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")
    mp4_file = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_file.exists():
        pytest.skip("Test combined MP4 file not generated")

    result = MediaService.probe_file(mp4_file)
    assert result.filename == "test_combined.mp4"
    assert result.has_video is True
    assert result.has_audio is True
    assert result.is_supported is True
    assert len(result.video_streams) >= 1
    assert len(result.audio_streams) >= 1
    assert result.video_streams[0].width == 320
    assert result.video_streams[0].height == 240


def test_probe_path_with_spaces():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")
    space_file = TEST_MEDIA_DIR / "test audio with spaces.wav"
    if not space_file.exists():
        pytest.skip("Test space path file not generated")

    result = MediaService.probe_file(space_file)
    assert result.filename == "test audio with spaces.wav"
    assert result.has_audio is True


def test_malformed_unsupported_input():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")
    corrupt_file = PROJECT_ROOT / "data" / "temp" / "corrupt.txt"
    corrupt_file.parent.mkdir(parents=True, exist_ok=True)
    corrupt_file.write_text("This is not a valid media file.")

    try:
        with pytest.raises(Exception):
            MediaService.probe_file(corrupt_file)
    finally:
        if corrupt_file.exists():
            corrupt_file.unlink()


def test_api_media_status():
    response = client.get("/api/media/status")
    assert response.status_code == 200
    data = response.json()
    assert "ffmpeg_available" in data
    assert "ffprobe_available" in data
    assert "version" in data


def test_api_media_probe_endpoint():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")
    mp4_file = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_file.exists():
        pytest.skip("Test combined MP4 file not generated")

    with open(mp4_file, "rb") as f:
        response = client.post(
            "/api/media/probe",
            files={"file": ("uploaded_test.mp4", f, "video/mp4")}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "uploaded_test.mp4"
    assert data["has_video"] is True
    assert data["has_audio"] is True
    assert data["is_supported"] is True
    assert len(data["video_streams"]) == 1
