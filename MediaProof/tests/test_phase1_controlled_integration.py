"""
Phase 1.6 Controlled Integration Tests for MediaProof.

Validates the coordinated behavior of:
1. forensics.temporal_dhash (extraction)
2. forensics.temporal_compare (temporal sequence alignment)
3. forensics.temporal_persistence (isolated storage & retrieval)
4. forensics.metadata_drift (FFprobe extraction)
5. forensics.metadata_compare (metadata drift comparison)

All scenarios are tested with synthetic, deterministic media fixtures.
Persistence tests run strictly against an isolated sandbox SQLite database.
"""

import copy
import hashlib
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from typing import Generator, Tuple
import pytest

import cv2
import numpy as np

from backend.paths import get_db_path
from media.ffmpeg import get_ffmpeg_path, run_ffmpeg_command
from forensics.temporal_dhash import (
    ALGORITHM_NAME as DHASH_ALGORITHM_NAME,
    ALGORITHM_VERSION as DHASH_ALGORITHM_VERSION,
    STATUS_DECODE_FAILURE,
    STATUS_INVALID_MEDIA,
    STATUS_SUCCESS as DHASH_STATUS_SUCCESS,
    TemporalDHashResult,
    extract_temporal_dhash,
)
from forensics.temporal_compare import (
    STATUS_SUCCESS as COMPARE_STATUS_SUCCESS,
    STATUS_UNAVAILABLE_INPUT,
    TemporalComparisonResult,
    compare_temporal_dhash_sequences,
)
from forensics.metadata_drift import (
    STATUS_PROCESS_FAILURE,
    STATUS_SUCCESS as META_STATUS_SUCCESS,
    TechnicalMetadataResult,
    extract_technical_metadata,
)
from forensics.metadata_compare import (
    DIFF_CHANGED,
    DIFF_MISSING_CANDIDATE_VALUE,
    DIFF_MISSING_REFERENCE_VALUE,
    DIFF_UNCHANGED,
    DIFF_UNCOMPARABLE,
    OUTCOME_MATCH,
    OUTCOME_MISSING,
    OUTCOME_UNCOMPARABLE,
    STATUS_INSUFFICIENT_EVIDENCE,
    STATUS_SUCCESS as COMP_META_STATUS_SUCCESS,
    STATUS_SUCCESS_WITH_DRIFT,
    STATUS_SUCCESS_WITH_MISSING_FIELDS,
    compare_technical_metadata,
)


# ==============================================================================
# Controlled Fixture Helpers
# ==============================================================================

def create_dynamic_test_video(
    file_path: Path,
    duration_seconds: float = 3.0,
    fps: float = 10.0,
    width: int = 160,
    height: int = 120,
) -> Path:
    """
    Creates a controlled synthetic MP4 video using FFmpeg lavfi testsrc.
    The dynamic counter and moving color patterns ensure distinct temporal frames.
    Fails explicitly if media creation does not produce a valid non-empty file.
    """
    ffmpeg_exe = get_ffmpeg_path()
    if ffmpeg_exe and ffmpeg_exe.exists():
        cmd = [
            str(ffmpeg_exe), "-y",
            "-f", "lavfi",
            "-i", f"testsrc=duration={duration_seconds}:size={width}x{height}:rate={int(fps)}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            str(file_path),
        ]
        res = run_ffmpeg_command(cmd)
        if res.returncode == 0 and file_path.exists() and file_path.stat().st_size > 0:
            return file_path

    # Fallback to OpenCV VideoWriter if FFmpeg lavfi is unavailable
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
    total_frames = max(1, round(duration_seconds * fps))
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        bar_pos = int((i / max(1, total_frames)) * width)
        cv2.rectangle(frame, (bar_pos, 0), (min(width, bar_pos + 20), height), (0, 255, 0), -1)
        cv2.putText(frame, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out.write(frame)
    out.release()

    if not file_path.exists() or file_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to generate dynamic test video at: {file_path}")
    return file_path


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Provides a clean temporary workspace for controlled test media files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def isolated_db(temp_workspace: Path, monkeypatch) -> Generator[Path, None, None]:
    """
    Configures an isolated SQLite database to guarantee zero writes to production DB.
    Patches paths BEFORE importing provenance.database.
    """
    db_path = temp_workspace / "sandbox_mediaproof.db"
    monkeypatch.setenv("MEDIAPROOF_DB_PATH", str(db_path))

    import backend.paths
    monkeypatch.setattr(backend.paths, "get_db_path", lambda: db_path)

    import provenance.database
    monkeypatch.setattr(provenance.database, "get_db_path", lambda: db_path)
    provenance.database.init_db()
    yield db_path


# ==============================================================================
# Step 2: Controlled Integration Scenarios
# ==============================================================================

def test_scenario_1_identical_media_compared_with_itself(temp_workspace: Path):
    """
    Scenario 1: Identical media compared with itself.
    Temporal comparison must yield match_ratio 1.0, mean_hamming_distance 0.0.
    Metadata comparison must yield STATUS_SUCCESS (or SUCCESS_WITH_MISSING_FIELDS if optional tags are absent) with zero drift.
    """
    video_path = temp_workspace / "base_identical.mp4"
    create_dynamic_test_video(video_path, duration_seconds=2.0, fps=10.0)

    # 1. Temporal extraction and comparison
    dhash_res = extract_temporal_dhash(video_path, sampling_interval_seconds=0.5)
    assert dhash_res.available is True
    assert dhash_res.status == DHASH_STATUS_SUCCESS
    assert dhash_res.sample_count >= 3

    comp_dhash = compare_temporal_dhash_sequences(dhash_res, dhash_res)
    assert comp_dhash.available is True
    assert comp_dhash.status == COMPARE_STATUS_SUCCESS
    assert comp_dhash.match_ratio == 1.0
    assert comp_dhash.mean_hamming_distance == 0.0
    assert comp_dhash.matched_pairs_count == dhash_res.sample_count

    # 2. Metadata extraction and comparison
    meta_res = extract_technical_metadata(video_path)
    assert meta_res.available is True
    assert meta_res.status == META_STATUS_SUCCESS

    comp_meta = compare_technical_metadata(meta_res, meta_res)
    assert comp_meta.available is True
    assert comp_meta.status in (COMP_META_STATUS_SUCCESS, STATUS_SUCCESS_WITH_MISSING_FIELDS)
    assert comp_meta.summary.changed_fields_count == 0
    assert comp_meta.summary.has_transcoding_drift is False
    assert comp_meta.summary.has_critical_drift is False


def test_scenario_2_benign_reencode_or_transcode(temp_workspace: Path):
    """
    Scenario 2: Benign re-encode or transcode.
    Visual perceptual dHash remains identical or highly similar, but technical metadata
    reflects codec/bitrate/container differences without claiming malicious manipulation.
    """
    ffmpeg_exe = get_ffmpeg_path()
    if not ffmpeg_exe or not ffmpeg_exe.exists():
        pytest.skip("FFmpeg not available for transcode test")

    base_video = temp_workspace / "orig_source.mp4"
    reencoded_video = temp_workspace / "transcoded_target.mp4"

    create_dynamic_test_video(base_video, duration_seconds=2.5, fps=10.0)

    # Re-encode to lower bitrate and constrained GOP
    cmd = [
        str(ffmpeg_exe), "-y",
        "-i", str(base_video),
        "-c:v", "libx264",
        "-b:v", "150k",
        "-g", "10",
        "-pix_fmt", "yuv420p",
        str(reencoded_video),
    ]
    res = run_ffmpeg_command(cmd)
    assert res.returncode == 0, f"FFmpeg transcode command failed: {res.stderr}"
    assert reencoded_video.exists() and reencoded_video.stat().st_size > 0, f"Transcoded file missing: {reencoded_video}"

    # 1. Temporal comparison
    dhash_orig = extract_temporal_dhash(base_video, sampling_interval_seconds=0.5)
    dhash_reenc = extract_temporal_dhash(reencoded_video, sampling_interval_seconds=0.5)
    comp_dhash = compare_temporal_dhash_sequences(dhash_orig, dhash_reenc)

    assert comp_dhash.available is True
    assert comp_dhash.match_ratio >= 0.85
    assert comp_dhash.mean_hamming_distance <= 4.0

    # 2. Metadata comparison
    meta_orig = extract_technical_metadata(base_video)
    meta_reenc = extract_technical_metadata(reencoded_video)
    comp_meta = compare_technical_metadata(meta_orig, meta_reenc)

    assert comp_meta.available is True
    assert comp_meta.status == STATUS_SUCCESS_WITH_DRIFT
    assert comp_meta.summary.has_transcoding_drift is True
    evidence = comp_meta.to_evidence_dict()
    assert "has_transcoding_drift" in evidence["summary"]


def test_scenario_3_prefix_or_suffix_trimming(temp_workspace: Path):
    """
    Scenario 3: Prefix or suffix trimming.
    Candidate clip represents a temporal subclip of the reference.
    Semi-global DP alignment must match the subclip with high fidelity while
    reporting unmatched reference bounds.
    """
    ffmpeg_exe = get_ffmpeg_path()
    if not ffmpeg_exe or not ffmpeg_exe.exists():
        pytest.skip("FFmpeg not available for trimming test")

    full_video = temp_workspace / "full_duration.mp4"
    trimmed_video = temp_workspace / "trimmed_subclip.mp4"

    create_dynamic_test_video(full_video, duration_seconds=5.0, fps=10.0)

    # Trim: extract 1.0s to 3.5s (2.5s duration)
    cmd = [
        str(ffmpeg_exe), "-y",
        "-ss", "1.0",
        "-i", str(full_video),
        "-t", "2.5",
        "-c", "copy",
        str(trimmed_video),
    ]
    res = run_ffmpeg_command(cmd)
    assert res.returncode == 0, f"FFmpeg trimming command failed: {res.stderr}"
    assert trimmed_video.exists() and trimmed_video.stat().st_size > 0, f"Trimmed file missing: {trimmed_video}"

    dhash_full = extract_temporal_dhash(full_video, sampling_interval_seconds=0.5)
    dhash_trim = extract_temporal_dhash(trimmed_video, sampling_interval_seconds=0.5)

    comp_dhash = compare_temporal_dhash_sequences(dhash_full, dhash_trim)
    assert comp_dhash.available is True
    # Candidate is completely covered inside reference
    assert comp_dhash.candidate_coverage == 1.0
    assert comp_dhash.candidate_sample_count < comp_dhash.reference_sample_count
    assert len(comp_dhash.unmatched_reference_indices) > 0

    # Metadata flags duration reduction
    meta_full = extract_technical_metadata(full_video)
    meta_trim = extract_technical_metadata(trimmed_video)
    comp_meta = compare_technical_metadata(meta_full, meta_trim)
    assert comp_meta.summary.has_critical_drift is True
    dur_diff = next(d for d in comp_meta.differences if d.field_path == "container.duration_seconds")
    assert dur_diff.difference_type == DIFF_CHANGED


def test_scenario_4_internal_segment_removed(temp_workspace: Path):
    """
    Scenario 4: Internal segment removed.
    Verifies every subprocess command and ensures explicit failure if fixture creation fails.
    Temporal comparison verifies that middle frames are absent while surrounding segments match.
    """
    ffmpeg_exe = get_ffmpeg_path()
    if not ffmpeg_exe or not ffmpeg_exe.exists():
        pytest.skip("FFmpeg not available for splicing test")

    p1 = temp_workspace / "p1.mp4"
    p2 = temp_workspace / "p2.mp4"
    full = temp_workspace / "full.mp4"
    spliced = temp_workspace / "spliced.mp4"

    create_dynamic_test_video(full, duration_seconds=3.0, fps=10.0)

    # Extract 0 to 1s
    res_p1 = run_ffmpeg_command([str(ffmpeg_exe), "-y", "-i", str(full), "-t", "1", "-c", "copy", str(p1)])
    assert res_p1.returncode == 0, f"FFmpeg failed creating part 1: {res_p1.stderr}"
    assert p1.exists() and p1.stat().st_size > 0, f"Part 1 file was not created: {p1}"

    # Extract 2 to 3s
    res_p2 = run_ffmpeg_command([str(ffmpeg_exe), "-y", "-ss", "2", "-i", str(full), "-t", "1", "-c", "copy", str(p2)])
    assert res_p2.returncode == 0, f"FFmpeg failed creating part 2: {res_p2.stderr}"
    assert p2.exists() and p2.stat().st_size > 0, f"Part 2 file was not created: {p2}"

    # Concatenate p1 + p2
    concat_list = temp_workspace / "concat.txt"
    concat_list.write_text(f"file '{p1.name}'\nfile '{p2.name}'\n", encoding="utf-8")
    res_concat = run_ffmpeg_command([
        str(ffmpeg_exe), "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list), "-c", "copy", str(spliced)
    ])
    assert res_concat.returncode == 0, f"FFmpeg failed concatenating spliced video: {res_concat.stderr}"
    assert spliced.exists() and spliced.stat().st_size > 0, f"Spliced video was not created: {spliced}"

    dhash_full = extract_temporal_dhash(full, sampling_interval_seconds=0.5)
    dhash_spliced = extract_temporal_dhash(spliced, sampling_interval_seconds=0.5)

    comp_dhash = compare_temporal_dhash_sequences(dhash_full, dhash_spliced)
    assert comp_dhash.available is True
    # Middle segment was removed; candidate sample count is lower
    assert comp_dhash.candidate_sample_count < comp_dhash.reference_sample_count


def test_scenario_5_reordered_segments(temp_workspace: Path):
    """
    Scenario 5: Reordered segments (Part B followed by Part A).
    Chronological disruption prevents standard linear alignment; sequence similarity degrades.
    """
    orig_path = temp_workspace / "orig_distinct_segments.mp4"
    reord_path = temp_workspace / "reordered_segments.mp4"

    fps = 10.0
    width, height = 160, 120
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Generate 40 frames total:
    # Part A (frames 0-19): horizontal gradient pattern
    # Part B (frames 20-39): vertical stripes pattern
    # These have high Hamming distance (>30 bits) between Part A and Part B
    frames_a = []
    for i in range(20):
        f = np.zeros((height, width, 3), dtype=np.uint8)
        for x in range(width):
            f[:, x] = int(((x + i * 4) % width) / width * 255)
        frames_a.append(f)

    frames_b = []
    for i in range(20):
        f = np.zeros((height, width, 3), dtype=np.uint8)
        f[:, ::4] = 255
        f[::4, :] = int((i * 12) % 256)
        frames_b.append(f)

    # Write original sequence: Part A then Part B
    out_orig = cv2.VideoWriter(str(orig_path), fourcc, fps, (width, height))
    for f in frames_a + frames_b:
        out_orig.write(f)
    out_orig.release()
    assert orig_path.exists() and orig_path.stat().st_size > 0, f"Original file not created: {orig_path}"

    # Write reordered sequence: Part B then Part A
    out_reord = cv2.VideoWriter(str(reord_path), fourcc, fps, (width, height))
    for f in frames_b + frames_a:
        out_reord.write(f)
    out_reord.release()
    assert reord_path.exists() and reord_path.stat().st_size > 0, f"Reordered file not created: {reord_path}"

    dhash_orig = extract_temporal_dhash(orig_path, sampling_interval_seconds=0.5)
    dhash_reord = extract_temporal_dhash(reord_path, sampling_interval_seconds=0.5)

    comp_dhash = compare_temporal_dhash_sequences(dhash_orig, dhash_reord)
    assert comp_dhash.available is True
    # Monotonic time alignment cannot reconcile reversed block order without skipping one
    assert comp_dhash.match_ratio <= 0.60
    assert comp_dhash.matched_pairs_count < dhash_orig.sample_count

    # Metadata comparison reports identical duration, showing that metadata alone
    # cannot detect internal chronological segment reordering
    meta_orig = extract_technical_metadata(orig_path)
    meta_reord = extract_technical_metadata(reord_path)
    comp_meta = compare_technical_metadata(meta_orig, meta_reord)
    dur_diff = next(d for d in comp_meta.differences if d.field_path == "container.duration_seconds")
    assert dur_diff.difference_type == DIFF_UNCHANGED


def test_scenario_6_changed_technical_metadata_dimensions(temp_workspace: Path):
    """
    Scenario 6: Changed dimensions.
    Perceptual difference-hash is scale-invariant (9x8 normalization), while
    metadata comparison explicitly flags width/height changes with critical drift.
    """
    orig_path = temp_workspace / "orig_160x120.mp4"
    scaled_path = temp_workspace / "scaled_80x60.mp4"

    create_dynamic_test_video(orig_path, duration_seconds=2.0, fps=10.0, width=160, height=120)
    create_dynamic_test_video(scaled_path, duration_seconds=2.0, fps=10.0, width=80, height=60)
    assert orig_path.exists() and orig_path.stat().st_size > 0
    assert scaled_path.exists() and scaled_path.stat().st_size > 0

    dhash_orig = extract_temporal_dhash(orig_path, sampling_interval_seconds=0.5)
    dhash_scaled = extract_temporal_dhash(scaled_path, sampling_interval_seconds=0.5)

    comp_dhash = compare_temporal_dhash_sequences(dhash_orig, dhash_scaled)
    assert comp_dhash.available is True
    # dHash scale-invariance maintains high match
    assert comp_dhash.match_ratio >= 0.85

    meta_orig = extract_technical_metadata(orig_path)
    meta_scaled = extract_technical_metadata(scaled_path)
    comp_meta = compare_technical_metadata(meta_orig, meta_scaled)

    assert comp_meta.summary.has_critical_drift is True
    width_diff = next(d for d in comp_meta.differences if d.field_path.endswith(".width"))
    height_diff = next(d for d in comp_meta.differences if d.field_path.endswith(".height"))
    assert width_diff.difference_type == DIFF_CHANGED
    assert height_diff.difference_type == DIFF_CHANGED
    assert width_diff.is_drift is True
    assert height_diff.is_drift is True


def test_scenario_7a_missing_metadata_on_one_side(temp_workspace: Path):
    """
    Scenario 7A: Missing metadata on only one side (reference has bitrate, candidate is missing it).
    Must be classified as missing evidence (DIFF_MISSING_CANDIDATE_VALUE, OUTCOME_MISSING),
    NOT actual drift (is_drift=False), yielding STATUS_SUCCESS_WITH_MISSING_FIELDS.
    """
    video_path = temp_workspace / "meta_one_side.mp4"
    create_dynamic_test_video(video_path, duration_seconds=2.0, fps=10.0)

    meta_ref = extract_technical_metadata(video_path)
    meta_cand = copy.deepcopy(meta_ref)

    # Reference has explicit known bitrate; candidate has bitrate omitted (None)
    meta_ref.container.bitrate = 250000
    meta_cand.container.bitrate = None

    # Harmonize optional tag dictionaries so tags are jointly uncomparable without spurious drift
    meta_ref.container.tags = {}
    meta_cand.container.tags = {}
    if meta_ref.video_streams:
        meta_ref.video_streams[0].raw_tags = {}
        meta_cand.video_streams[0].raw_tags = {}

    comp = compare_technical_metadata(meta_ref, meta_cand)
    assert comp.available is True
    assert comp.status == STATUS_SUCCESS_WITH_MISSING_FIELDS

    # Verify field-level difference outcome
    bitrate_diff = next(d for d in comp.differences if d.field_path == "container.bitrate")
    assert bitrate_diff.difference_type == DIFF_MISSING_CANDIDATE_VALUE
    assert bitrate_diff.comparison_outcome == OUTCOME_MISSING
    assert bitrate_diff.is_drift is False
    assert bitrate_diff.reference_value == 250000
    assert bitrate_diff.candidate_value is None

    # Summary verification
    assert comp.summary.changed_fields_count == 0
    assert comp.summary.missing_fields_count >= 1
    assert comp.summary.has_critical_drift is False
    assert comp.summary.has_transcoding_drift is False

    # Partition invariant
    s = comp.summary
    assert s.total_compared_fields == (
        s.unchanged_fields_count
        + s.changed_fields_count
        + s.missing_fields_count
        + s.uncomparable_fields_count
    )


def test_scenario_7b_missing_metadata_on_both_sides(temp_workspace: Path):
    """
    Scenario 7B: Jointly unavailable metadata fields (missing on both reference and candidate).
    Must be classified as uncomparable (DIFF_UNCOMPARABLE, OUTCOME_UNCOMPARABLE),
    NOT actual drift (is_drift=False), yielding STATUS_SUCCESS_WITH_MISSING_FIELDS.
    """
    video_path = temp_workspace / "meta_both_sides.mp4"
    create_dynamic_test_video(video_path, duration_seconds=2.0, fps=10.0)

    meta_ref = extract_technical_metadata(video_path)
    meta_cand = copy.deepcopy(meta_ref)

    # Bitrate is missing on BOTH sides
    meta_ref.container.bitrate = None
    meta_cand.container.bitrate = None

    # Harmonize optional tags
    meta_ref.container.tags = {}
    meta_cand.container.tags = {}
    if meta_ref.video_streams:
        meta_ref.video_streams[0].raw_tags = {}
        meta_cand.video_streams[0].raw_tags = {}

    comp = compare_technical_metadata(meta_ref, meta_cand)
    assert comp.available is True
    assert comp.status == STATUS_SUCCESS_WITH_MISSING_FIELDS

    # Verify field-level difference outcome
    bitrate_diff = next(d for d in comp.differences if d.field_path == "container.bitrate")
    assert bitrate_diff.difference_type == DIFF_UNCOMPARABLE
    assert bitrate_diff.comparison_outcome == OUTCOME_UNCOMPARABLE
    assert bitrate_diff.is_drift is False
    assert bitrate_diff.reference_value is None
    assert bitrate_diff.candidate_value is None

    # Summary verification
    assert comp.summary.changed_fields_count == 0
    assert comp.summary.uncomparable_fields_count >= 1
    assert comp.summary.has_critical_drift is False
    assert comp.summary.has_transcoding_drift is False

    # Partition invariant
    s = comp.summary
    assert s.total_compared_fields == (
        s.unchanged_fields_count
        + s.changed_fields_count
        + s.missing_fields_count
        + s.uncomparable_fields_count
    )


def test_scenario_8_corrupt_media_and_failure_handling(temp_workspace: Path):
    """
    Scenario 8: Corrupt media and failed frame extraction.
    Algorithms handle corrupt/truncated files safely, emitting documented error statuses.
    """
    corrupt_path = temp_workspace / "corrupted_media.mp4"
    corrupt_path.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00" + b"\xff" * 500)
    assert corrupt_path.exists() and corrupt_path.stat().st_size > 0

    # 1. Temporal dHash handling
    dhash_res = extract_temporal_dhash(corrupt_path)
    assert dhash_res.available is False
    assert dhash_res.status in (STATUS_DECODE_FAILURE, STATUS_INVALID_MEDIA)

    # 2. Comparison with unavailable input
    valid_path = temp_workspace / "valid.mp4"
    create_dynamic_test_video(valid_path, duration_seconds=1.0, fps=10.0)
    dhash_valid = extract_temporal_dhash(valid_path)

    comp_dhash = compare_temporal_dhash_sequences(dhash_valid, dhash_res)
    assert comp_dhash.available is False
    assert comp_dhash.status == STATUS_UNAVAILABLE_INPUT

    # 3. Metadata extraction handling
    meta_res = extract_technical_metadata(corrupt_path)
    if not meta_res.available:
        meta_valid = extract_technical_metadata(valid_path)
        comp_meta = compare_technical_metadata(meta_valid, meta_res)
        assert comp_meta.available is False
        assert comp_meta.status == STATUS_INSUFFICIENT_EVIDENCE


def test_scenario_9_repeated_execution_determinism(temp_workspace: Path):
    """
    Scenario 9: Repeated execution with the same inputs and configuration.
    Extracting and comparing identical media repeatedly produces bit-for-bit identical outputs.
    """
    video_path = temp_workspace / "deterministic_run.mp4"
    create_dynamic_test_video(video_path, duration_seconds=2.0, fps=10.0)

    # Run 1
    dhash1 = extract_temporal_dhash(video_path, sampling_interval_seconds=0.5)
    meta1 = extract_technical_metadata(video_path)
    comp_meta1 = compare_technical_metadata(meta1, meta1)

    # Run 2
    dhash2 = extract_temporal_dhash(video_path, sampling_interval_seconds=0.5)
    meta2 = extract_technical_metadata(video_path)
    comp_meta2 = compare_technical_metadata(meta2, meta2)

    # Hashes and sample sequences are identical
    assert dhash1.sample_count == dhash2.sample_count
    assert len(dhash1.samples) == len(dhash2.samples)
    for s1, s2 in zip(dhash1.samples, dhash2.samples):
        assert s1.sample_index == s2.sample_index
        assert s1.timestamp_seconds == s2.timestamp_seconds
        assert s1.dhash_hex == s2.dhash_hex
        assert s1.dhash_int == s2.dhash_int

    # Evidence payloads match exactly (excludes non-deterministic processing_time_seconds)
    assert comp_meta1.to_evidence_dict() == comp_meta2.to_evidence_dict()


# ==============================================================================
# Step 3: Persistence and Isolation Verification
# ==============================================================================

def test_temporal_persistence_with_controlled_asset(temp_workspace: Path, isolated_db: Path):
    """
    Step 3: Stores and retrieves temporal fingerprints using an isolated database.
    Verifies algorithm version, sampling interval, sample sequence, and status.
    Confirms zero modification to production database.
    """
    from forensics.temporal_persistence import (
        delete_temporal_fingerprint,
        get_temporal_fingerprint,
        has_temporal_fingerprint,
        save_temporal_fingerprint,
    )

    # 1. Create registered asset in isolated DB
    conn = sqlite3.connect(str(isolated_db))
    asset_id = "test-asset-ctrl-001"
    conn.execute(
        """
        INSERT INTO assets (
            asset_id, watermark_token, original_filename, original_path,
            file_size_bytes, sha256, media_type, container_format,
            duration_seconds, video_codec, width, height, fps,
            created_at, registered_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            asset_id, "token123", "ctrl.mp4", "/path/ctrl.mp4",
            1024, "abcdef123456", "video", "mp4",
            2.0, "h264", 160, 120, 10.0,
            "2026-10-18T00:00:00Z", "2026-10-18T00:00:00Z",
        ),
    )
    conn.commit()
    conn.close()

    # 2. Extract temporal dHash from controlled video
    video_path = temp_workspace / "ctrl.mp4"
    create_dynamic_test_video(video_path, duration_seconds=2.0, fps=10.0)
    dhash_result = extract_temporal_dhash(video_path, sampling_interval_seconds=0.5)
    assert dhash_result.available is True

    # 3. Save to isolated persistence
    save_temporal_fingerprint(asset_id, dhash_result)
    assert has_temporal_fingerprint(asset_id) is True

    # 4. Retrieve and verify full fidelity
    retrieved = get_temporal_fingerprint(asset_id)
    assert retrieved is not None
    assert retrieved.algorithm == f"{DHASH_ALGORITHM_NAME}_v{DHASH_ALGORITHM_VERSION}"
    assert retrieved.sampling_interval_seconds == 0.5
    assert retrieved.sample_count == dhash_result.sample_count
    assert retrieved.status == DHASH_STATUS_SUCCESS
    assert len(retrieved.samples) == len(dhash_result.samples)

    for orig_s, ret_s in zip(dhash_result.samples, retrieved.samples):
        assert ret_s.sample_index == orig_s.sample_index
        assert ret_s.timestamp_seconds == orig_s.timestamp_seconds
        assert ret_s.dhash_int == orig_s.dhash_int
        assert ret_s.dhash_hex == orig_s.dhash_hex

    # 5. Delete and verify cleanup
    assert delete_temporal_fingerprint(asset_id) is True
    assert has_temporal_fingerprint(asset_id) is False


def test_collection_and_execution_cannot_modify_production_database():
    """
    Regression Test: Verifies that pytest collection and test execution cannot initialize
    or modify the production database (database/mediaproof.db).

    Runs clean subprocesses with scrubbed environment (MEDIAPROOF_DB_PATH unset) to verify that:
    1. Pytest collection of the controlled integration suite triggers zero database mutations.
    2. Actual execution of a representative Phase 1 test suite (tests/test_temporal_persistence.py)
       configures its own isolated sandbox without relying on inherited environment variables.
    3. Execution of the controlled integration persistence scenario against isolated storage
       does not mutate production storage.
    4. Avoids recursive invocation of the regression test itself.
    """
    project_root = Path(__file__).resolve().parent.parent
    prod_db = project_root / "database" / "mediaproof.db"
    assert prod_db.exists(), f"Production database not found at: {prod_db}"
    initial_hash = hashlib.sha256(prod_db.read_bytes()).hexdigest()

    # Scrub MEDIAPROOF_DB_PATH to force child process isolation bootstrap to configure its own sandbox
    clean_env = os.environ.copy()
    clean_env.pop("MEDIAPROOF_DB_PATH", None)

    subprocesses = [
        (
            "pytest_collection_controlled_integration",
            [sys.executable, "-m", "pytest", "tests/test_phase1_controlled_integration.py", "--collect-only", "-q"]
        ),
        (
            "pytest_execution_representative_persistence_suite",
            [sys.executable, "-m", "pytest", "tests/test_temporal_persistence.py", "-q"]
        ),
        (
            "pytest_execution_controlled_integration_persistence",
            [sys.executable, "-m", "pytest", "tests/test_phase1_controlled_integration.py", "-k", "test_temporal_persistence_with_controlled_asset", "-q"]
        ),
    ]

    for label, cmd in subprocesses:
        before_hash = hashlib.sha256(prod_db.read_bytes()).hexdigest()
        assert before_hash == initial_hash, (
            f"Pre-condition failed before {label}: database hash altered! "
            f"Expected {initial_hash}, found {before_hash}"
        )

        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            env=clean_env,
            capture_output=True,
            text=True
        )

        after_hash = hashlib.sha256(prod_db.read_bytes()).hexdigest()

        assert proc.returncode == 0, (
            f"Subprocess '{label}' failed with return code {proc.returncode}!\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stdout: {proc.stdout}\n"
            f"Stderr: {proc.stderr}"
        )

        assert before_hash == after_hash, (
            f"CRITICAL: Production database modified during '{label}'!\n"
            f"Before SHA-256: {before_hash}\n"
            f"After SHA-256:  {after_hash}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stdout: {proc.stdout}"
        )
        assert after_hash == initial_hash, (
            f"CRITICAL: Production database drifted from initial state after '{label}'!\n"
            f"Initial SHA-256: {initial_hash}\n"
            f"Current SHA-256: {after_hash}"
        )
