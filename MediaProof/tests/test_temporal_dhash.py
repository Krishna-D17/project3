"""
Tests for Deterministic Temporal Difference-Hash (dHash) Extraction (Phase 1.1).

Covers:
1. Deterministic hashes for controlled synthetic frames.
2. Correct 64-bit output and documented bit ordering (MSB to LSB).
3. Identical input/configuration producing identical timestamp/hash sequences.
4. Configurable sampling intervals.
5. Short videos and videos with fewer frames than expected.
6. Missing input file handling.
7. Corrupt or unreadable video handling.
8. Decoder unavailable or decoder failure handling.
9. Invalid sampling intervals (zero, negative, NaN, Inf, non-numeric).
10. No usable sampled frames handling.
11. Repeatability across repeated extraction calls.
"""

import math
from pathlib import Path
import pytest
import numpy as np
import cv2

from forensics.temporal_dhash import (
    ALGORITHM_NAME,
    ALGORITHM_VERSION,
    DHASH_BITS,
    DHASH_HEIGHT,
    DHASH_WIDTH,
    STATUS_DECODE_FAILURE,
    STATUS_INVALID_INTERVAL,
    STATUS_INVALID_MEDIA,
    STATUS_SUCCESS,
    STATUS_UNAVAILABLE_DECODER,
    STATUS_ZERO_SAMPLES,
    TemporalDHashResult,
    TemporalDHashSample,
    compute_frame_dhash,
    extract_temporal_dhash,
)


def create_synthetic_video(file_path: Path, duration_seconds: float = 3.0, fps: float = 10.0) -> Path:
    """Helper to generate a deterministic synthetic MP4 video fixture on CPU."""
    total_frames = max(1, round(duration_seconds * fps))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(file_path), fourcc, fps, (64, 64))

    for i in range(total_frames):
        # Create a deterministic pattern: vertical split that moves over time
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        boundary = int((i / max(1, total_frames)) * 64)
        frame[:, :boundary] = 255
        out.write(frame)

    out.release()
    return file_path


# ==============================================================================
# 1. Deterministic hashes for controlled synthetic frames
# ==============================================================================

def test_dhash_synthetic_controlled_frames():
    # 1. All left pixels greater than right pixels across all 8 rows -> 64 ones = 0xffffffffffffffff
    img_all_ones = np.tile(np.linspace(255, 0, 9, dtype=np.uint8), (8, 1))
    h_int, h_hex = compute_frame_dhash(img_all_ones)
    assert h_int == 0xFFFFFFFFFFFFFFFF
    assert h_hex == "ffffffffffffffff"

    # 2. All left pixels less than right pixels across all 8 rows -> 64 zeros = 0x0000000000000000
    img_all_zeros = np.tile(np.linspace(0, 255, 9, dtype=np.uint8), (8, 1))
    h_int, h_hex = compute_frame_dhash(img_all_zeros)
    assert h_int == 0
    assert h_hex == "0000000000000000"

    # 3. Alternating rows
    img_alt = np.zeros((8, 9), dtype=np.uint8)
    for r in range(8):
        if r % 2 == 0:
            img_alt[r, :] = np.linspace(255, 0, 9, dtype=np.uint8)  # 8 ones
        else:
            img_alt[r, :] = np.linspace(0, 255, 9, dtype=np.uint8)  # 8 zeros
    h_int, h_hex = compute_frame_dhash(img_alt)
    # Row 0: 0xff, Row 1: 0x00, Row 2: 0xff, Row 3: 0x00... -> ff00ff00ff00ff00
    assert h_hex == "ff00ff00ff00ff00"


# ==============================================================================
# 2. Correct 64-bit output and documented bit ordering
# ==============================================================================

def test_dhash_64bit_output_and_bit_ordering():
    # Test MSB bit (row 0, col 0): left > right only at r=0, c=0
    # Baseline all 100, set r=0, c=0 to 200 -> only (0, 0) > (0, 1) is True
    # Expected: bit 63 is 1, all other 63 bits are 0 -> 0x8000000000000000
    img_msb = np.full((8, 9), 100, dtype=np.uint8)
    img_msb[0, 0] = 200
    h_int, h_hex = compute_frame_dhash(img_msb)
    assert len(h_hex) == 16
    assert h_int == (1 << 63)
    assert h_hex == "8000000000000000"

    # Test LSB bit (row 7, col 7): left > right only at r=7, c=7
    # Baseline all 100, set r=7, c=7 to 200 -> only (7, 7) > (7, 8) is True
    # Expected: bit 0 is 1, all other 63 bits are 0 -> 0x0000000000000001
    img_lsb = np.full((8, 9), 100, dtype=np.uint8)
    img_lsb[7, 7] = 200
    h_int, h_hex = compute_frame_dhash(img_lsb)
    assert len(h_hex) == 16
    assert h_int == 1
    assert h_hex == "0000000000000001"


# ==============================================================================
# 3. Identical input/configuration producing identical sequences
# ==============================================================================

def test_identical_input_produces_identical_sequences(tmp_path):
    vid_path = create_synthetic_video(tmp_path / "identical_test.mp4", duration_seconds=2.0, fps=10.0)

    res1 = extract_temporal_dhash(vid_path, sampling_interval_seconds=1.0)
    res2 = extract_temporal_dhash(vid_path, sampling_interval_seconds=1.0)

    assert res1.available is True
    assert res2.available is True
    assert res1.status == STATUS_SUCCESS
    assert res2.status == STATUS_SUCCESS
    assert res1.sample_count == res2.sample_count
    assert res1.sample_count > 0

    for s1, s2 in zip(res1.samples, res2.samples):
        assert s1.sample_index == s2.sample_index
        assert s1.timestamp_seconds == s2.timestamp_seconds
        assert s1.frame_index == s2.frame_index
        assert s1.dhash_hex == s2.dhash_hex
        assert s1.dhash_int == s2.dhash_int


# ==============================================================================
# 4. Configurable sampling interval
# ==============================================================================

def test_configurable_sampling_interval(tmp_path):
    # 4-second video at 10 fps (40 frames)
    vid_path = create_synthetic_video(tmp_path / "interval_test.mp4", duration_seconds=4.0, fps=10.0)

    # 1.0 second interval -> t = 0.0, 1.0, 2.0, 3.0 (4 samples)
    res_1s = extract_temporal_dhash(vid_path, sampling_interval_seconds=1.0)
    assert res_1s.available is True
    assert res_1s.sample_count == 4
    assert [s.timestamp_seconds for s in res_1s.samples] == [0.0, 1.0, 2.0, 3.0]

    # 0.5 second interval -> t = 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 (8 samples)
    res_05s = extract_temporal_dhash(vid_path, sampling_interval_seconds=0.5)
    assert res_05s.available is True
    assert res_05s.sample_count == 8
    assert [s.timestamp_seconds for s in res_05s.samples] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

    # 2.0 second interval -> t = 0.0, 2.0 (2 samples)
    res_2s = extract_temporal_dhash(vid_path, sampling_interval_seconds=2.0)
    assert res_2s.available is True
    assert res_2s.sample_count == 2
    assert [s.timestamp_seconds for s in res_2s.samples] == [0.0, 2.0]


# ==============================================================================
# 5. Short videos and videos with fewer frames than expected
# ==============================================================================

def test_short_video_handling(tmp_path):
    # Short video: 0.3s (3 frames at 10 fps), shorter than default 1.0s interval
    short_vid = create_synthetic_video(tmp_path / "short.mp4", duration_seconds=0.3, fps=10.0)
    res = extract_temporal_dhash(short_vid, sampling_interval_seconds=1.0)

    assert res.available is True
    assert res.status == STATUS_SUCCESS
    assert res.sample_count == 1
    assert res.samples[0].timestamp_seconds == 0.0
    assert res.samples[0].sample_index == 0

    # Extremely short: 1 single frame (0.1s at 10 fps)
    one_frame_vid = create_synthetic_video(tmp_path / "one_frame.mp4", duration_seconds=0.1, fps=10.0)
    res_one = extract_temporal_dhash(one_frame_vid, sampling_interval_seconds=1.0)
    assert res_one.available is True
    assert res_one.sample_count == 1
    assert res_one.samples[0].timestamp_seconds == 0.0


# ==============================================================================
# 6. Missing input file
# ==============================================================================

def test_missing_input_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.mp4"
    res = extract_temporal_dhash(missing_path)

    assert res.available is False
    assert res.status == STATUS_INVALID_MEDIA
    assert res.sample_count == 0
    assert "error" in res.details
    assert "not found" in res.details["error"].lower()


# ==============================================================================
# 7. Corrupt or unreadable video
# ==============================================================================

def test_corrupt_video_handling(tmp_path):
    corrupt_file = tmp_path / "corrupted.mp4"
    corrupt_file.write_bytes(b"RANDOM_CORRUPTED_BYTES_NOT_A_VALID_CONTAINER_STREAM")

    res = extract_temporal_dhash(corrupt_file)

    assert res.available is False
    assert res.status in [STATUS_INVALID_MEDIA, STATUS_ZERO_SAMPLES]
    assert res.sample_count == 0
    assert "error" in res.details


# ==============================================================================
# 8. Decoder unavailable or decoder failure
# ==============================================================================

def test_decoder_unavailable_handling(monkeypatch, tmp_path):
    vid_path = create_synthetic_video(tmp_path / "dec_test.mp4", duration_seconds=1.0, fps=10.0)

    # Simulate cv2 missing
    monkeypatch.setattr("forensics.temporal_dhash.HAS_CV2", False)
    monkeypatch.setattr("forensics.temporal_dhash.cv2", None)

    res = extract_temporal_dhash(vid_path)
    assert res.available is False
    assert res.status == STATUS_UNAVAILABLE_DECODER
    assert "OpenCV" in res.details.get("error", "")

    # Simulate VideoCapture exception
    monkeypatch.undo()
    def mock_bad_vc(*args, **kwargs):
        raise RuntimeError("Decoder driver crashed")

    monkeypatch.setattr(cv2, "VideoCapture", mock_bad_vc)
    res_exc = extract_temporal_dhash(vid_path)
    assert res_exc.available is False
    assert res_exc.status in [STATUS_INVALID_MEDIA, STATUS_DECODE_FAILURE]


# ==============================================================================
# 9. Invalid sampling intervals
# ==============================================================================

def test_invalid_sampling_intervals(tmp_path):
    vid_path = create_synthetic_video(tmp_path / "invalid_interval.mp4", duration_seconds=1.0, fps=10.0)

    for bad_interval in [0.0, 0, -1.0, -10, float("nan"), float("inf"), float("-inf"), "invalid"]:
        res = extract_temporal_dhash(vid_path, sampling_interval_seconds=bad_interval)
        assert res.available is False
        assert res.status == STATUS_INVALID_INTERVAL
        assert res.sample_count == 0
        assert "positive" in res.details.get("error", "")


# ==============================================================================
# 10. No usable sampled frames
# ==============================================================================

def test_no_usable_sampled_frames(monkeypatch, tmp_path):
    vid_path = create_synthetic_video(tmp_path / "no_usable.mp4", duration_seconds=1.0, fps=10.0)

    # Mock VideoCapture.read to always return (False, None)
    class MockEmptyCap:
        def isOpened(self):
            return True
        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 10.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 10.0
            return 0.0
        def set(self, prop, val):
            return True
        def read(self):
            return False, None
        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: MockEmptyCap())

    res = extract_temporal_dhash(vid_path)
    assert res.available is False
    assert res.status == STATUS_ZERO_SAMPLES
    assert res.sample_count == 0
    assert "zero usable frames" in res.details.get("error", "").lower()


# ==============================================================================
# 11. Repeatability across repeated calls
# ==============================================================================

def test_repeatability_across_repeated_calls(tmp_path):
    vid_path = create_synthetic_video(tmp_path / "repeat_test.mp4", duration_seconds=2.0, fps=10.0)

    results = [extract_temporal_dhash(vid_path, sampling_interval_seconds=0.5) for _ in range(5)]

    first_hashes = [s.dhash_hex for s in results[0].samples]
    first_timestamps = [s.timestamp_seconds for s in results[0].samples]

    for r in results[1:]:
        assert r.available is True
        assert r.status == STATUS_SUCCESS
        assert r.sample_count == len(first_hashes)
        assert [s.dhash_hex for s in r.samples] == first_hashes
        assert [s.timestamp_seconds for s in r.samples] == first_timestamps
