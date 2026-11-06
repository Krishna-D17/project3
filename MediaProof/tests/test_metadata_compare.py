"""
Tests for Technical Metadata Drift Comparison (Phase 1.5 Corrective Pass).

Covers:
1. Identical complete metadata produces STATUS_SUCCESS and zero drift.
2. Actual differences with no missing values produce STATUS_SUCCESS_WITH_DRIFT.
3. Missing optional values with no observed differences produce STATUS_SUCCESS_WITH_MISSING_FIELDS.
4. Missing values alongside actual differences produce STATUS_SUCCESS_WITH_DRIFT.
5. Unavailable extraction results produce STATUS_INSUFFICIENT_EVIDENCE.
6. Unresolved stream correspondences produce STATUS_INSUFFICIENT_EVIDENCE.
7. Missing required container metadata produces STATUS_INSUFFICIENT_EVIDENCE.
8. Container format changes (compatible aliases vs distinct formats).
9. Duration, bitrate, and file-size differences.
10. Video resolution and codec changes.
11. Frame-rate and color-property differences.
12. Audio codec, sample-rate, and channel-layout differences.
13. Added and removed streams.
14. Reordered streams with distinguishable attributes aligned deterministically.
15. Multiple reference streams preferring the same candidate, where global assignment resolves the conflict.
16. Genuinely ambiguous stream assignment with multiple equally optimal solutions.
17. Candidate stream with no plausible correspondence (zero affinity).
18. Unequal numbers of reference and candidate streams.
19. Repeated comparisons producing bit-for-bit identical stream match records.
20. Missing optional values and jointly unavailable properties categorized as UNCOMPARABLE.
21. Tags added, removed, and modified.
22. Strict argument validation for duration_tolerance_seconds (negative, NaN, Inf, non-numeric).
23. Strict argument validation for fps_tolerance (negative, NaN, Inf, non-numeric).
24. Malformed nested metadata structures rejected safely with STATUS_INVALID_INPUT.
25. Deterministic evidence serialization via to_evidence_dict() across repeated runs.
26. Input objects remain 100% unmodified after comparison.
27. Benign transcoding differences emit factual evidence without final MediaProof verdicts.
28. Summary counts strictly partition all compared field records:
    total_compared == unchanged + changed + missing + uncomparable.
"""

import copy
from typing import Dict, List, Optional
import pytest

from forensics.metadata_compare import (
    DIFF_ADDED,
    DIFF_CHANGED,
    DIFF_MISSING_CANDIDATE_VALUE,
    DIFF_MISSING_REFERENCE_VALUE,
    DIFF_REMOVED,
    DIFF_UNCOMPARABLE,
    DIFF_UNCHANGED,
    OUTCOME_DRIFT,
    OUTCOME_MATCH,
    OUTCOME_MISSING,
    OUTCOME_UNCOMPARABLE,
    STATUS_INSUFFICIENT_EVIDENCE,
    STATUS_INVALID_ARGUMENTS,
    STATUS_INVALID_INPUT,
    STATUS_SUCCESS,
    STATUS_SUCCESS_WITH_DRIFT,
    STATUS_SUCCESS_WITH_MISSING_FIELDS,
    STREAM_ADDED,
    STREAM_AMBIGUOUS,
    STREAM_MATCHED,
    STREAM_REMOVED,
    MetadataComparisonResult,
    compare_format_names,
    compare_technical_metadata,
    find_all_optimal_matchings,
    match_streams_deterministically,
)
from forensics.metadata_drift import (
    AudioStreamMetadata,
    ContainerMetadata,
    TechnicalMetadataResult,
    VideoStreamMetadata,
)


def create_sample_result(
    file_path: str = "media.mp4",
    available: bool = True,
    status: str = "SUCCESS",
    format_name: Optional[str] = "mov,mp4,m4a,3gp,3g2,mj2",
    duration: Optional[float] = 10.0,
    bitrate: Optional[int] = 2000000,
    start_time: Optional[float] = 0.0,
    size_bytes: Optional[int] = 2500000,
    encoder: Optional[str] = "Lavf60.3.100",
    container_tags: Optional[Dict[str, str]] = None,
    video_streams: Optional[List[VideoStreamMetadata]] = None,
    audio_streams: Optional[List[AudioStreamMetadata]] = None,
) -> TechnicalMetadataResult:
    """Helper to construct complete, deterministic TechnicalMetadataResult objects."""
    c_meta = ContainerMetadata(
        format_name=format_name,
        duration_seconds=duration,
        bitrate=bitrate,
        start_time_seconds=start_time,
        size_bytes=size_bytes,
        encoder=encoder,
        tags=container_tags if container_tags is not None else {"major_brand": "isom", "encoder": "Lavf60.3.100"},
    )

    if video_streams is None:
        v_streams = [
            VideoStreamMetadata(
                stream_index=0,
                codec_name="h264",
                profile="High",
                level=40,
                pixel_format="yuv420p",
                width=1920,
                height=1080,
                sample_aspect_ratio="1:1",
                display_aspect_ratio="16:9",
                avg_frame_rate=30.0,
                r_frame_rate=30.0,
                color_range="tv",
                color_space="bt709",
                color_transfer="bt709",
                color_primaries="bt709",
                frame_count=300,
                time_base="1/30000",
                duration_seconds=10.0,
                bitrate=1800000,
                raw_tags={"handler_name": "VideoHandler"},
            )
        ]
    else:
        v_streams = video_streams

    if audio_streams is None:
        a_streams = [
            AudioStreamMetadata(
                stream_index=1,
                codec_name="aac",
                sample_rate=48000,
                channels=2,
                channel_layout="stereo",
                duration_seconds=10.0,
                bitrate=192000,
                raw_tags={"language": "eng"},
            )
        ]
    else:
        a_streams = audio_streams

    return TechnicalMetadataResult(
        available=available,
        status=status,
        file_path=file_path,
        container=c_meta,
        video_streams=v_streams,
        audio_streams=a_streams,
        video_stream_count=len(v_streams),
        audio_stream_count=len(a_streams),
        ffprobe_version="ffprobe 7.0.2",
    )


# ==============================================================================
# 1. Status Semantics Tests (Correct Comparison Status Contract)
# ==============================================================================

def test_identical_complete_metadata_produces_status_success():
    """Identical complete metadata produces STATUS_SUCCESS with zero drift."""
    ref = create_sample_result("ref.mp4")
    cand = create_sample_result("cand.mp4")

    res = compare_technical_metadata(ref, cand)

    assert res.available is True
    assert res.status == STATUS_SUCCESS
    assert res.summary.unchanged_fields_count == res.summary.total_compared_fields
    assert res.summary.changed_fields_count == 0
    assert res.summary.missing_fields_count == 0
    assert res.summary.uncomparable_fields_count == 0
    assert res.summary.has_critical_drift is False
    assert res.summary.has_transcoding_drift is False
    assert all(not d.is_drift for d in res.differences)


def test_actual_differences_with_no_missing_values_produces_status_success_with_drift():
    """Actual differences with complete metadata produce STATUS_SUCCESS_WITH_DRIFT."""
    ref = create_sample_result()
    cand = create_sample_result(bitrate=1200000)  # Changed bitrate

    res = compare_technical_metadata(ref, cand)

    assert res.available is True
    assert res.status == STATUS_SUCCESS_WITH_DRIFT
    assert res.summary.changed_fields_count >= 1
    assert res.summary.missing_fields_count == 0
    assert res.summary.has_transcoding_drift is True


def test_missing_optional_values_with_no_observed_differences_produces_success_with_missing_fields():
    """Missing optional values with no differences produce STATUS_SUCCESS_WITH_MISSING_FIELDS."""
    ref = create_sample_result()
    cand = create_sample_result()
    cand.video_streams[0].profile = None  # Missing optional candidate profile

    res = compare_technical_metadata(ref, cand)

    assert res.available is True
    assert res.status == STATUS_SUCCESS_WITH_MISSING_FIELDS
    assert res.summary.changed_fields_count == 0
    assert res.summary.missing_fields_count >= 1
    # Crucial: missing optional values alone MUST NOT trigger STATUS_SUCCESS_WITH_DRIFT
    assert res.status != STATUS_SUCCESS_WITH_DRIFT


def test_missing_values_alongside_actual_differences_produces_success_with_drift():
    """Missing optional values alongside actual differences produce STATUS_SUCCESS_WITH_DRIFT."""
    ref = create_sample_result()
    cand = create_sample_result(duration=14.0)  # Actual duration drift
    cand.video_streams[0].profile = None        # Missing optional candidate profile

    res = compare_technical_metadata(ref, cand)

    assert res.available is True
    assert res.status == STATUS_SUCCESS_WITH_DRIFT
    assert res.summary.changed_fields_count >= 1
    assert res.summary.missing_fields_count >= 1


def test_unavailable_extraction_results_produce_insufficient_evidence():
    """Unavailable metadata extraction results produce STATUS_INSUFFICIENT_EVIDENCE."""
    valid = create_sample_result()
    unavail = create_sample_result(available=False, status="FFPROBE_UNAVAILABLE")

    # Reference unavailable
    res_ref = compare_technical_metadata(unavail, valid)
    assert res_ref.available is False
    assert res_ref.status == STATUS_INSUFFICIENT_EVIDENCE
    assert "Reference metadata unavailable" in res_ref.details["error"]

    # Candidate unavailable
    res_cand = compare_technical_metadata(valid, unavail)
    assert res_cand.available is False
    assert res_cand.status == STATUS_INSUFFICIENT_EVIDENCE
    assert "Candidate metadata unavailable" in res_cand.details["error"]


def test_unresolved_stream_correspondence_produces_insufficient_evidence():
    """Ambiguous/unresolved stream correspondences produce STATUS_INSUFFICIENT_EVIDENCE."""
    ref_v = VideoStreamMetadata(stream_index=0, codec_name="h264", width=1280, height=720)
    ref = create_sample_result(video_streams=[ref_v])

    # Candidate has dual identical video streams that cannot be uniquely resolved
    c_v1 = VideoStreamMetadata(stream_index=3, codec_name="h264", width=1280, height=720)
    c_v2 = VideoStreamMetadata(stream_index=4, codec_name="h264", width=1280, height=720)
    cand = create_sample_result(video_streams=[c_v1, c_v2])

    res = compare_technical_metadata(ref, cand)

    assert res.available is True
    assert res.status == STATUS_INSUFFICIENT_EVIDENCE
    assert any(m.match_status == STREAM_AMBIGUOUS for m in res.stream_matches)
    assert res.summary.uncomparable_fields_count >= 1


def test_missing_required_container_metadata_produces_insufficient_evidence():
    """Missing required container metadata produces STATUS_INSUFFICIENT_EVIDENCE."""
    ref = create_sample_result()
    cand = create_sample_result()
    cand.container = None  # Missing container

    res = compare_technical_metadata(ref, cand)

    assert res.available is False
    assert res.status == STATUS_INSUFFICIENT_EVIDENCE
    assert "Candidate container metadata is missing" in res.details["error"]


# ==============================================================================
# 2. Reproducibility & Accurate Serialization Tests
# ==============================================================================

def test_deterministic_evidence_serialization_via_to_evidence_dict():
    """Repeated comparisons produce identical serialized evidence via to_evidence_dict()."""
    ref = create_sample_result("ref_video.mp4", bitrate=4000000)
    cand = create_sample_result("cand_video.mp4", bitrate=2500000)

    res1 = compare_technical_metadata(ref, cand)
    res2 = compare_technical_metadata(ref, cand)

    # to_evidence_dict() excludes runtime measurements and is bit-for-bit reproducible
    dict1 = res1.to_evidence_dict()
    dict2 = res2.to_evidence_dict()

    assert dict1 == dict2
    assert "processing_time_seconds" not in dict1
    assert "processing_time_seconds" not in dict2

    # to_dict(include_runtime_measurements=True) preserves runtime timing
    dict_full = res1.to_dict(include_runtime_measurements=True)
    assert "processing_time_seconds" in dict_full
    assert isinstance(dict_full["processing_time_seconds"], float)


def test_inputs_remain_unmodified():
    """Reference and candidate input objects are not mutated during comparison."""
    ref = create_sample_result("ref.mp4")
    cand = create_sample_result("cand.mp4", duration=12.5)

    ref_dict_before = ref.to_dict()
    cand_dict_before = cand.to_dict()

    _ = compare_technical_metadata(ref, cand)

    assert ref.to_dict() == ref_dict_before
    assert cand.to_dict() == cand_dict_before


# ==============================================================================
# 3. Global Stream Alignment & Ambiguity Resolution Tests
# ==============================================================================

def test_reordered_streams_with_distinguishable_attributes():
    """Reordered streams are matched by attributes and mapped deterministically."""
    # Reference: Stream 0 is HD h264, Stream 1 is SD vp9
    v_ref0 = VideoStreamMetadata(stream_index=0, codec_name="h264", width=1920, height=1080)
    v_ref1 = VideoStreamMetadata(stream_index=1, codec_name="vp9", width=640, height=360)
    ref = create_sample_result(video_streams=[v_ref0, v_ref1])

    # Candidate: Stream 0 is SD vp9, Stream 1 is HD h264
    v_cand0 = VideoStreamMetadata(stream_index=0, codec_name="vp9", width=640, height=360)
    v_cand1 = VideoStreamMetadata(stream_index=1, codec_name="h264", width=1920, height=1080)
    cand = create_sample_result(video_streams=[v_cand0, v_cand1])

    res = compare_technical_metadata(ref, cand)

    # Both streams matched one-to-one
    assert res.summary.added_streams_count == 0
    assert res.summary.removed_streams_count == 0

    hd_match = next(m for m in res.stream_matches if m.reference_stream_index == 0)
    assert hd_match.candidate_stream_index == 1
    assert hd_match.match_status == STREAM_MATCHED

    sd_match = next(m for m in res.stream_matches if m.reference_stream_index == 1)
    assert sd_match.candidate_stream_index == 0
    assert sd_match.match_status == STREAM_MATCHED


def test_multiple_reference_streams_preferring_same_candidate_resolved_globally():
    """Global assignment prevents greedy conflict where both reference streams prefer candidate 0."""
    # Ref 0: h264 1920x1080 (High affinity to Cand 0 and Cand 1)
    # Ref 1: h264 1280x720  (Prefers Cand 1, low affinity to Cand 0)
    v_ref0 = VideoStreamMetadata(stream_index=0, codec_name="h264", width=1920, height=1080)
    v_ref1 = VideoStreamMetadata(stream_index=1, codec_name="h264", width=1280, height=720)
    ref = create_sample_result(video_streams=[v_ref0, v_ref1])

    # Cand 0: 1920x1080, Cand 1: 1280x720
    v_cand0 = VideoStreamMetadata(stream_index=0, codec_name="h264", width=1920, height=1080)
    v_cand1 = VideoStreamMetadata(stream_index=1, codec_name="h264", width=1280, height=720)
    cand = create_sample_result(video_streams=[v_cand0, v_cand1])

    res = compare_technical_metadata(ref, cand)

    m0 = next(m for m in res.stream_matches if m.reference_stream_index == 0)
    m1 = next(m for m in res.stream_matches if m.reference_stream_index == 1)

    assert m0.candidate_stream_index == 0
    assert m1.candidate_stream_index == 1
    assert m0.match_status == STREAM_MATCHED
    assert m1.match_status == STREAM_MATCHED


def test_genuinely_ambiguous_assignment_reports_ambiguity():
    """Equal optimal assignments report STREAM_AMBIGUOUS without arbitrary guessing."""
    a_ref = AudioStreamMetadata(stream_index=0, codec_name="aac", sample_rate=48000, channels=2)
    ref = create_sample_result(audio_streams=[a_ref])

    # Candidate has two completely identical audio tracks with disjoint indices
    a_c1 = AudioStreamMetadata(stream_index=2, codec_name="aac", sample_rate=48000, channels=2)
    a_c2 = AudioStreamMetadata(stream_index=3, codec_name="aac", sample_rate=48000, channels=2)
    cand = create_sample_result(audio_streams=[a_c1, a_c2])

    res = compare_technical_metadata(ref, cand)

    assert res.status == STATUS_INSUFFICIENT_EVIDENCE
    ambiguous = [m for m in res.stream_matches if m.match_status == STREAM_AMBIGUOUS]
    assert len(ambiguous) == 1
    assert ambiguous[0].reference_stream_index == 0
    assert ambiguous[0].candidate_stream_index is None


def test_candidate_stream_with_no_plausible_correspondence_reported_as_added():
    """Candidate stream with zero affinity (no plausible attributes) is reported as added."""
    v_ref = VideoStreamMetadata(stream_index=0, codec_name="h264", width=1920, height=1080)
    ref = create_sample_result(video_streams=[v_ref], audio_streams=[])

    # Candidate has valid video stream + totally dissimilar extra audio stream
    v_cand = VideoStreamMetadata(stream_index=0, codec_name="h264", width=1920, height=1080)
    a_cand = AudioStreamMetadata(stream_index=1, codec_name="flac", sample_rate=96000, channels=8)
    cand = create_sample_result(video_streams=[v_cand], audio_streams=[a_cand])

    res = compare_technical_metadata(ref, cand)

    assert res.summary.added_streams_count == 1
    added_audio = next(m for m in res.stream_matches if m.candidate_stream_index == 1 and m.stream_type == "audio")
    assert added_audio.match_status == STREAM_ADDED
    assert added_audio.reference_stream_index is None


def test_unequal_numbers_of_streams():
    """Unequal stream counts report STREAM_REMOVED or STREAM_ADDED accurately."""
    v_ref0 = VideoStreamMetadata(stream_index=0, codec_name="h264", width=1920, height=1080)
    v_ref1 = VideoStreamMetadata(stream_index=1, codec_name="vp9", width=640, height=360)
    ref = create_sample_result(video_streams=[v_ref0, v_ref1])

    # Candidate dropped stream 1
    v_cand0 = VideoStreamMetadata(stream_index=0, codec_name="h264", width=1920, height=1080)
    cand = create_sample_result(video_streams=[v_cand0])

    res = compare_technical_metadata(ref, cand)

    assert res.summary.removed_streams_count == 1
    rem = next(m for m in res.stream_matches if m.reference_stream_index == 1)
    assert rem.match_status == STREAM_REMOVED


def test_repeated_runs_produce_identical_stream_matches():
    """Repeated comparisons produce identical stream matches."""
    ref = create_sample_result()
    cand = create_sample_result()

    res1 = compare_technical_metadata(ref, cand)
    res2 = compare_technical_metadata(ref, cand)

    matches1 = [m.to_dict() for m in res1.stream_matches]
    matches2 = [m.to_dict() for m in res2.stream_matches]

    assert matches1 == matches2


# ==============================================================================
# 4. Public Argument & Malformed Input Validation Tests
# ==============================================================================

@pytest.mark.parametrize(
    "invalid_tol,err_substr",
    [
        (-0.01, "cannot be negative"),
        (float("nan"), "cannot be NaN"),
        (float("inf"), "cannot be NaN or Infinite"),
        (float("-inf"), "cannot be NaN or Infinite"),
        ("invalid", "must be a non-negative finite number"),
        (None, "must be a non-negative finite number"),
        (True, "must be a non-negative finite number"),
    ],
)
def test_invalid_duration_tolerance_rejected(invalid_tol, err_substr):
    """Negative, NaN, infinite, and non-numeric duration tolerances are rejected."""
    ref = create_sample_result()
    cand = create_sample_result()

    res = compare_technical_metadata(ref, cand, duration_tolerance_seconds=invalid_tol)  # type: ignore

    assert res.available is False
    assert res.status == STATUS_INVALID_INPUT
    assert err_substr in res.details["error"]


@pytest.mark.parametrize(
    "invalid_tol,err_substr",
    [
        (-0.001, "cannot be negative"),
        (float("nan"), "cannot be NaN"),
        (float("inf"), "cannot be NaN or Infinite"),
        ("bad_fps", "must be a non-negative finite number"),
        (None, "must be a non-negative finite number"),
        (False, "must be a non-negative finite number"),
    ],
)
def test_invalid_fps_tolerance_rejected(invalid_tol, err_substr):
    """Negative, NaN, infinite, and non-numeric fps tolerances are rejected."""
    ref = create_sample_result()
    cand = create_sample_result()

    res = compare_technical_metadata(ref, cand, fps_tolerance=invalid_tol)  # type: ignore

    assert res.available is False
    assert res.status == STATUS_INVALID_INPUT
    assert err_substr in res.details["error"]


def test_malformed_nested_metadata_rejected_safely():
    """Structurally malformed metadata objects return STATUS_INVALID_INPUT with structured error."""
    ref = create_sample_result()
    cand = create_sample_result()

    # Corrupt video_streams with a non-list
    cand.video_streams = "not_a_list"  # type: ignore
    res_bad_list = compare_technical_metadata(ref, cand)
    assert res_bad_list.available is False
    assert res_bad_list.status == STATUS_INVALID_INPUT
    assert "video_streams must be a list" in res_bad_list.details["error"]

    # Corrupt video_streams item with an invalid object
    cand.video_streams = ["not_a_video_stream"]  # type: ignore
    res_bad_item = compare_technical_metadata(ref, cand)
    assert res_bad_item.available is False
    assert res_bad_item.status == STATUS_INVALID_INPUT
    assert "VideoStreamMetadata" in res_bad_item.details["error"]


def test_invalid_argument_types():
    """Non-TechnicalMetadataResult inputs return STATUS_INVALID_INPUT."""
    valid = create_sample_result()

    res_none = compare_technical_metadata(None, valid)  # type: ignore
    assert res_none.available is False
    assert res_none.status == STATUS_INVALID_INPUT

    res_str = compare_technical_metadata(valid, "not_a_result")  # type: ignore
    assert res_str.available is False
    assert res_str.status == STATUS_INVALID_INPUT


# ==============================================================================
# 5. Jointly Unavailable & Missing Optional Values Tests
# ==============================================================================

def test_jointly_unavailable_optional_values_categorized_as_uncomparable():
    """When a property is absent in both inputs, it is categorized as UNCOMPARABLE, not MATCH."""
    ref = create_sample_result()
    cand = create_sample_result()

    # Set optional color properties to None in both
    ref.video_streams[0].color_transfer = None
    cand.video_streams[0].color_transfer = None

    res = compare_technical_metadata(ref, cand)

    diff = next(d for d in res.differences if "color_transfer" in d.field_path)
    assert diff.difference_type == DIFF_UNCOMPARABLE
    assert diff.comparison_outcome == OUTCOME_UNCOMPARABLE
    assert diff.is_drift is False
    assert "unavailable in both" in diff.explanation

    # Jointly unavailable values are counted under uncomparable, not unchanged
    assert res.summary.uncomparable_fields_count >= 1
    assert res.summary.jointly_unavailable_fields_count >= 1


def test_missing_optional_values_categorization():
    """One-sided missing optional values are classified as MISSING_REFERENCE or MISSING_CANDIDATE."""
    ref = create_sample_result()
    ref.video_streams[0].profile = "High"
    ref.video_streams[0].level = None  # Missing in ref

    cand = create_sample_result()
    cand.video_streams[0].profile = None  # Missing in candidate
    cand.video_streams[0].level = 41     # Present in candidate

    res = compare_technical_metadata(ref, cand)

    profile_diff = next(d for d in res.differences if "profile" in d.field_path)
    level_diff = next(d for d in res.differences if "level" in d.field_path)

    assert profile_diff.difference_type == DIFF_MISSING_CANDIDATE_VALUE
    assert profile_diff.comparison_outcome == OUTCOME_MISSING

    assert level_diff.difference_type == DIFF_MISSING_REFERENCE_VALUE
    assert level_diff.comparison_outcome == OUTCOME_MISSING

    assert res.summary.missing_fields_count >= 2


# ==============================================================================
# 6. Container Format & Transcoding Tests
# ==============================================================================

def test_container_format_exact_change():
    """Distinct container formats are categorized as CHANGED with explanation."""
    ref = create_sample_result(format_name="mp4")
    cand = create_sample_result(format_name="matroska,webm")

    res = compare_technical_metadata(ref, cand)

    assert res.available is True
    fmt_diff = next(d for d in res.differences if d.field_path == "container.format_name")
    assert fmt_diff.difference_type == DIFF_CHANGED
    assert fmt_diff.is_drift is True
    assert "matroska,webm" in fmt_diff.explanation


def test_container_format_compatible_alias_not_classified_as_drift():
    """Compatible container format aliases within same family (e.g. mov,mp4... vs mp4) are not drift."""
    ref = create_sample_result(format_name="mov,mp4,m4a,3gp,3g2,mj2")
    cand = create_sample_result(format_name="mp4")

    res = compare_technical_metadata(ref, cand)

    assert res.available is True
    assert res.status == STATUS_SUCCESS
    fmt_diff = next(d for d in res.differences if d.field_path == "container.format_name")
    assert fmt_diff.difference_type == DIFF_UNCHANGED
    assert fmt_diff.comparison_outcome == OUTCOME_MATCH
    assert fmt_diff.is_drift is False
    assert "compatible known alias family" in fmt_diff.explanation
    assert res.summary.has_transcoding_drift is False
    assert res.summary.changed_fields_count == 0


@pytest.mark.parametrize(
    "ref_fmt,cand_fmt,family_name",
    [
        ("mov,mp4,m4a,3gp,3g2,mj2", "mp4", "mp4_mov"),
        ("mp4", "mov", "mp4_mov"),
        ("matroska,webm", "webm", "matroska_webm"),
        ("matroska,webm", "mkv", "matroska_webm"),
        ("ogg", "ogv", "ogg"),
        ("wav", "wave", "wav"),
    ],
)
def test_explicitly_supported_format_aliases(ref_fmt, cand_fmt, family_name):
    """Explicitly supported format alias families match without being classified as drift."""
    diff_type, outcome, explanation, is_drift = compare_format_names(ref_fmt, cand_fmt)
    assert diff_type == DIFF_UNCHANGED
    assert outcome == OUTCOME_MATCH
    assert is_drift is False
    assert family_name in explanation


@pytest.mark.parametrize(
    "ref_fmt,cand_fmt",
    [
        ("mov,mp4,m4a,3gp,3g2,mj2", "matroska,webm"),
        ("mp4", "avi"),
        ("ogg", "mp3"),
        ("flv", "mp4"),
    ],
)
def test_genuinely_different_container_formats_reported_as_drift(ref_fmt, cand_fmt):
    """Genuinely different container formats are classified as CHANGED drift."""
    diff_type, outcome, explanation, is_drift = compare_format_names(ref_fmt, cand_fmt)
    assert diff_type == DIFF_CHANGED
    assert outcome == OUTCOME_DRIFT
    assert is_drift is True
    assert "changed from" in explanation


@pytest.mark.parametrize(
    "ref_fmt,cand_fmt",
    [
        ("mp4", "isom"),
        ("mp4", "mp41"),
        ("mp4", "mp42"),
        ("mov,mp4,m4a,3gp,3g2,mj2", "isom"),
        ("mov,mp4,m4a,3gp,3g2,mj2", "mp42"),
        ("ogg", "opus"),
        ("ogv", "opus"),
    ],
)
def test_questionable_identifiers_not_equivalent_to_mp4_or_ogg(ref_fmt, cand_fmt):
    """MP4 brand metadata (isom, mp41, mp42) and codec names (opus) are not container format aliases."""
    diff_type, outcome, explanation, is_drift = compare_format_names(ref_fmt, cand_fmt)
    assert diff_type == DIFF_CHANGED
    assert outcome == OUTCOME_DRIFT
    assert is_drift is True
    assert "changed from" in explanation


def test_compare_technical_metadata_flags_brand_and_codec_as_drift():
    """compare_technical_metadata reports brand tag or codec mismatches as actual container drift."""
    # MP4 vs brand tag isom
    ref_mp4 = create_sample_result(format_name="mp4")
    cand_isom = create_sample_result(format_name="isom")
    res_isom = compare_technical_metadata(ref_mp4, cand_isom)

    assert res_isom.available is True
    assert res_isom.status == STATUS_SUCCESS_WITH_DRIFT
    assert res_isom.summary.has_transcoding_drift is True
    fmt_diff_isom = next(d for d in res_isom.differences if d.field_path == "container.format_name")
    assert fmt_diff_isom.difference_type == DIFF_CHANGED
    assert fmt_diff_isom.is_drift is True

    # Ogg vs codec name opus
    ref_ogg = create_sample_result(format_name="ogg")
    cand_opus = create_sample_result(format_name="opus")
    res_opus = compare_technical_metadata(ref_ogg, cand_opus)

    assert res_opus.available is True
    assert res_opus.status == STATUS_SUCCESS_WITH_DRIFT
    assert res_opus.summary.has_transcoding_drift is True
    fmt_diff_opus = next(d for d in res_opus.differences if d.field_path == "container.format_name")
    assert fmt_diff_opus.difference_type == DIFF_CHANGED
    assert fmt_diff_opus.is_drift is True


def test_arbitrary_shared_token_does_not_prove_equivalence():
    """Arbitrary format strings sharing an unwhitelisted token are NOT treated as equivalent."""
    ref_fmt = "custom_mux_a,shared_token"
    cand_fmt = "shared_token,custom_mux_b"

    diff_type, outcome, explanation, is_drift = compare_format_names(ref_fmt, cand_fmt)
    assert diff_type == DIFF_CHANGED
    assert outcome == OUTCOME_DRIFT
    assert is_drift is True


def test_missing_format_names_semantics():
    """Missing container format name on either side produces MISSING with is_drift=False."""
    # Missing candidate
    dt1, out1, exp1, drift1 = compare_format_names("mp4", None)
    assert dt1 == DIFF_MISSING_CANDIDATE_VALUE
    assert out1 == OUTCOME_MISSING
    assert drift1 is False

    # Missing reference
    dt2, out2, exp2, drift2 = compare_format_names(None, "mp4")
    assert dt2 == DIFF_MISSING_REFERENCE_VALUE
    assert out2 == OUTCOME_MISSING
    assert drift2 is False

    # Both missing
    dt3, out3, exp3, drift3 = compare_format_names(None, None)
    assert dt3 == DIFF_UNCOMPARABLE
    assert out3 == OUTCOME_UNCOMPARABLE
    assert drift3 is False



def test_duration_within_tolerance_is_unchanged():
    """Duration difference within tolerance is classified as UNCHANGED."""
    ref = create_sample_result(duration=10.000)
    cand = create_sample_result(duration=10.025)  # Delta 0.025s <= default 0.05s

    res = compare_technical_metadata(ref, cand)
    dur_diff = next(d for d in res.differences if d.field_path == "container.duration_seconds")
    assert dur_diff.difference_type == DIFF_UNCHANGED
    assert dur_diff.is_drift is False


def test_duration_exceeding_tolerance_is_drift():
    """Duration difference exceeding tolerance is classified as CHANGED."""
    ref = create_sample_result(duration=10.0)
    cand = create_sample_result(duration=10.25)  # Delta 0.25s > 0.05s

    res = compare_technical_metadata(ref, cand)
    dur_diff = next(d for d in res.differences if d.field_path == "container.duration_seconds")
    assert dur_diff.difference_type == DIFF_CHANGED
    assert dur_diff.is_drift is True


def test_large_duration_discrepancy_flags_critical():
    """Duration difference > 1.0s flags critical drift."""
    ref = create_sample_result(duration=60.0)
    cand = create_sample_result(duration=45.0)  # Subclip or 15s trim

    res = compare_technical_metadata(ref, cand)
    assert res.summary.has_critical_drift is True


def test_bitrate_and_size_differences():
    """Bitrate and size differences are reported with transcoding drift flag."""
    ref = create_sample_result(bitrate=5000000, size_bytes=10000000)
    cand = create_sample_result(bitrate=1500000, size_bytes=3000000)

    res = compare_technical_metadata(ref, cand)
    br_diff = next(d for d in res.differences if d.field_path == "container.bitrate")
    sz_diff = next(d for d in res.differences if d.field_path == "container.size_bytes")

    assert br_diff.difference_type == DIFF_CHANGED
    assert sz_diff.difference_type == DIFF_CHANGED
    assert res.summary.has_transcoding_drift is True


def test_video_resolution_downscaling():
    """Resolution downscaling is detected and flags critical drift."""
    ref = create_sample_result()
    v_cand = copy.deepcopy(ref.video_streams)
    v_cand[0].width = 1280
    v_cand[0].height = 720
    cand = create_sample_result(video_streams=v_cand)

    res = compare_technical_metadata(ref, cand)

    assert res.summary.has_critical_drift is True
    w_diff = next(d for d in res.differences if "width" in d.field_path)
    h_diff = next(d for d in res.differences if "height" in d.field_path)
    assert w_diff.difference_type == DIFF_CHANGED
    assert h_diff.difference_type == DIFF_CHANGED


def test_video_codec_change():
    """Video codec changes are detected."""
    ref = create_sample_result()
    v_cand = copy.deepcopy(ref.video_streams)
    v_cand[0].codec_name = "hevc"
    cand = create_sample_result(video_streams=v_cand)

    res = compare_technical_metadata(ref, cand)

    assert res.summary.has_transcoding_drift is True
    codec_diff = next(d for d in res.differences if "video" in d.field_path and "codec_name" in d.field_path)
    assert codec_diff.difference_type == DIFF_CHANGED
    assert codec_diff.reference_value == "h264"
    assert codec_diff.candidate_value == "hevc"


def test_framerate_differences():
    """Frame rate changes are detected."""
    ref = create_sample_result()
    v_cand = copy.deepcopy(ref.video_streams)
    v_cand[0].avg_frame_rate = 24.0  # 30.0 -> 24.0
    cand = create_sample_result(video_streams=v_cand)

    res = compare_technical_metadata(ref, cand)
    fps_diff = next(d for d in res.differences if "avg_frame_rate" in d.field_path)
    assert fps_diff.difference_type == DIFF_CHANGED
    assert fps_diff.is_drift is True


def test_color_properties_differences():
    """Color space and color range differences are detected."""
    ref = create_sample_result()
    v_cand = copy.deepcopy(ref.video_streams)
    v_cand[0].color_space = "bt2020nc"
    v_cand[0].color_range = "pc"
    cand = create_sample_result(video_streams=v_cand)

    res = compare_technical_metadata(ref, cand)
    cs_diff = next(d for d in res.differences if "color_space" in d.field_path)
    cr_diff = next(d for d in res.differences if "color_range" in d.field_path)
    assert cs_diff.difference_type == DIFF_CHANGED
    assert cr_diff.difference_type == DIFF_CHANGED


def test_audio_stream_differences():
    """Audio codec, sample rate, and channels differences are detected."""
    ref = create_sample_result()
    a_cand = copy.deepcopy(ref.audio_streams)
    a_cand[0].codec_name = "mp3"
    a_cand[0].sample_rate = 44100
    a_cand[0].channels = 1
    a_cand[0].channel_layout = "mono"
    cand = create_sample_result(audio_streams=a_cand)

    res = compare_technical_metadata(ref, cand)

    c_diff = next(d for d in res.differences if "audio" in d.field_path and "codec_name" in d.field_path)
    sr_diff = next(d for d in res.differences if "audio" in d.field_path and "sample_rate" in d.field_path)
    ch_diff = next(d for d in res.differences if "audio" in d.field_path and "channels" in d.field_path)

    assert c_diff.difference_type == DIFF_CHANGED
    assert sr_diff.difference_type == DIFF_CHANGED
    assert ch_diff.difference_type == DIFF_CHANGED


def test_added_and_removed_streams():
    """Added and removed streams update summary counts and flag critical drift."""
    ref = create_sample_result()
    v2 = VideoStreamMetadata(stream_index=1, codec_name="vp9", width=640, height=360)
    cand = create_sample_result(video_streams=[ref.video_streams[0], v2], audio_streams=[])

    res = compare_technical_metadata(ref, cand)

    assert res.summary.added_streams_count == 1
    assert res.summary.removed_streams_count == 1
    assert res.summary.has_critical_drift is True

    match_types = {m.match_status for m in res.stream_matches}
    assert STREAM_ADDED in match_types
    assert STREAM_REMOVED in match_types


def test_tags_added_and_removed():
    """Tags added, removed, and modified are categorized."""
    ref = create_sample_result(container_tags={"encoder": "Lavf", "title": "Original Film"})
    cand = create_sample_result(container_tags={"encoder": "Lavf", "custom_tag": "Watermarked"})

    res = compare_technical_metadata(ref, cand)

    title_diff = next(d for d in res.differences if d.field_path == "container.tags.title")
    custom_diff = next(d for d in res.differences if d.field_path == "container.tags.custom_tag")

    assert title_diff.difference_type == DIFF_REMOVED
    assert custom_diff.difference_type == DIFF_ADDED


def test_transcoding_drift_emits_factual_evidence_not_final_verdict():
    """Comparison results never emit final MediaProof decision classifications."""
    ref = create_sample_result(bitrate=5000000, encoder="Lavf60")
    cand = create_sample_result(bitrate=2000000, encoder="HandBrake")

    res = compare_technical_metadata(ref, cand)

    assert res.available is True
    assert res.status == STATUS_SUCCESS_WITH_DRIFT
    assert res.summary.has_transcoding_drift is True

    res_dict = res.to_dict()
    forbidden_verdicts = {
        "Verified Original",
        "Verified Derivative",
        "Potentially Manipulated",
        "Unknown Provenance",
        "TAMPERED",
        "DEEPFAKE",
    }
    res_str = str(res_dict)
    for verdict in forbidden_verdicts:
        assert verdict not in res_str
    for d in res.differences:
        assert d.difference_type not in forbidden_verdicts
        assert d.comparison_outcome not in forbidden_verdicts


def test_summary_counts_partition_all_differences():
    """Summary counts strictly partition all compared field differences."""
    ref = create_sample_result()
    cand = create_sample_result(bitrate=1000000, duration=11.0)
    cand.video_streams[0].profile = None  # Missing candidate value

    res = compare_technical_metadata(ref, cand)

    s = res.summary
    assert s.total_compared_fields == len(res.differences)
    assert s.total_compared_fields == (
        s.unchanged_fields_count
        + s.changed_fields_count
        + s.missing_fields_count
        + s.uncomparable_fields_count
    )
    assert s.total_compared_fields > 0
    assert s.changed_fields_count >= 2
    assert s.missing_fields_count >= 1


# ==============================================================================
# 8. Bounded Stream Matching Tests
# ==============================================================================

def test_bounded_matching_evaluations_limit_exceeded():
    """When search space exceeds max_evaluations, bound_exceeded is True without silent truncation."""
    # 6x6 fully tied affinity matrix creates 6! = 720 possible permutations
    n = 6
    matrix = [[50 for _ in range(n)] for _ in range(n)]

    max_score, max_pairs, mappings, bound_exceeded = find_all_optimal_matchings(
        ref_count=n,
        cand_count=n,
        affinity_matrix=matrix,
        max_evaluations=15,  # Artificially low bound to trigger early termination
    )

    assert bound_exceeded is True


def test_bounded_matching_optimal_mappings_limit_exceeded():
    """When number of optimal mappings hits max_optimal_mappings, bound_exceeded is True."""
    n = 5
    matrix = [[50 for _ in range(n)] for _ in range(n)]

    max_score, max_pairs, mappings, bound_exceeded = find_all_optimal_matchings(
        ref_count=n,
        cand_count=n,
        affinity_matrix=matrix,
        max_evaluations=10000,
        max_optimal_mappings=5,  # Capped at 5 optimal solutions
    )

    assert bound_exceeded is True
    assert len(mappings) <= 5


def test_bounded_stream_matching_reports_ambiguity_and_never_arbitrarily_matches():
    """When matching bound is exceeded, all streams are reported as AMBIGUOUS without guessing."""
    ref_streams = [
        VideoStreamMetadata(stream_index=i, codec_name="h264", width=1920, height=1080)
        for i in range(5)
    ]
    cand_streams = [
        VideoStreamMetadata(stream_index=i, codec_name="h264", width=1920, height=1080)
        for i in range(5)
    ]

    pairs, records = match_streams_deterministically(
        stream_type="video",
        ref_streams=ref_streams,
        cand_streams=cand_streams,
        max_evaluations=10,  # Bound exceeded
    )

    # Every reference stream must be reported AMBIGUOUS, not arbitrarily matched
    assert len(records) == 5
    for rec in records:
        assert rec.match_status == STREAM_AMBIGUOUS
        assert "computational work bound" in rec.explanation
        assert rec.candidate_stream_index is None

    for r, c, st in pairs:
        assert st == STREAM_AMBIGUOUS
        assert c is None


def test_bounded_stream_matching_deterministic_behavior_at_bound():
    """Repeated runs of bounded matching produce bit-for-bit identical results."""
    ref_streams = [
        VideoStreamMetadata(stream_index=i, codec_name="h264", width=1920, height=1080)
        for i in range(4)
    ]
    cand_streams = [
        VideoStreamMetadata(stream_index=i, codec_name="h264", width=1920, height=1080)
        for i in range(4)
    ]

    pairs1, records1 = match_streams_deterministically("video", ref_streams, cand_streams, max_evaluations=8)
    pairs2, records2 = match_streams_deterministically("video", ref_streams, cand_streams, max_evaluations=8)

    assert len(records1) == len(records2)
    for r1, r2 in zip(records1, records2):
        assert r1.to_dict() == r2.to_dict()


def test_stream_matching_edge_cases_and_unmatched():
    """Empty streams, unmatched streams with zero affinity, and unequal stream counts."""
    # 1. Empty streams
    pairs_empty, recs_empty = match_streams_deterministically("video", [], [])
    assert pairs_empty == []
    assert recs_empty == []

    # 2. Candidate streams only
    c_only = [VideoStreamMetadata(stream_index=0, codec_name="h264", width=1280, height=720)]
    pairs_add, recs_add = match_streams_deterministically("video", [], c_only)
    assert len(recs_add) == 1
    assert recs_add[0].match_status == STREAM_ADDED

    # 3. Reference streams only
    r_only = [VideoStreamMetadata(stream_index=0, codec_name="h264", width=1280, height=720)]
    pairs_rem, recs_rem = match_streams_deterministically("video", r_only, [])
    assert len(recs_rem) == 1
    assert recs_rem[0].match_status == STREAM_REMOVED

    # 4. Zero affinity (unmatched candidate stream)
    # Ref is 1080p h264, candidate has stream 0 matching and stream 1 as 0-affinity
    r_v = VideoStreamMetadata(stream_index=0, codec_name="h264", width=1920, height=1080)
    c_v0 = VideoStreamMetadata(stream_index=0, codec_name="h264", width=1920, height=1080)
    c_v1 = VideoStreamMetadata(stream_index=1, codec_name="", width=None, height=None)  # Zero affinity
    pairs_zero, recs_zero = match_streams_deterministically("video", [r_v], [c_v0, c_v1])

    matched = [r for r in recs_zero if r.match_status == STREAM_MATCHED]
    added = [r for r in recs_zero if r.match_status == STREAM_ADDED]
    assert len(matched) == 1
    assert matched[0].reference_stream_index == 0
    assert matched[0].candidate_stream_index == 0
    assert len(added) == 1
    assert added[0].candidate_stream_index == 1


# ==============================================================================
# 9. Consistent Missing-Value Semantics Tests
# ==============================================================================

def test_missing_optional_fields_without_drift_preserves_is_drift_false():
    """Missing optional fields do NOT set is_drift=True and produce SUCCESS_WITH_MISSING_FIELDS."""
    ref = create_sample_result()
    cand = create_sample_result()

    # Drop optional fields in candidate
    cand.container.bitrate = None
    cand.container.encoder = None
    cand.video_streams[0].profile = None
    cand.video_streams[0].level = None

    res = compare_technical_metadata(ref, cand)

    assert res.available is True
    assert res.status == STATUS_SUCCESS_WITH_MISSING_FIELDS
    assert res.summary.changed_fields_count == 0
    assert res.summary.missing_fields_count >= 4
    assert res.summary.has_transcoding_drift is False
    assert res.summary.has_critical_drift is False

    for d in res.differences:
        if d.difference_type in (DIFF_MISSING_REFERENCE_VALUE, DIFF_MISSING_CANDIDATE_VALUE, DIFF_UNCOMPARABLE):
            assert d.is_drift is False

    s = res.summary
    assert s.total_compared_fields == (
        s.unchanged_fields_count
        + s.changed_fields_count
        + s.missing_fields_count
        + s.uncomparable_fields_count
    )


def test_actual_drift_alongside_missing_fields():
    """Actual drift is detected and produces STATUS_SUCCESS_WITH_DRIFT even when optional fields are missing."""
    ref = create_sample_result(bitrate=5000000)
    cand = create_sample_result(bitrate=2500000)  # Actual bitrate drift
    cand.video_streams[0].profile = None          # Missing optional field

    res = compare_technical_metadata(ref, cand)

    assert res.available is True
    assert res.status == STATUS_SUCCESS_WITH_DRIFT
    assert res.summary.changed_fields_count >= 1
    assert res.summary.missing_fields_count >= 1
    assert res.summary.has_transcoding_drift is True

    # Bitrate difference is drift
    br_diff = next(d for d in res.differences if d.field_path == "container.bitrate")
    assert br_diff.difference_type == DIFF_CHANGED
    assert br_diff.is_drift is True

    # Profile difference is missing, NOT drift
    prof_diff = next(d for d in res.differences if "profile" in d.field_path)
    assert prof_diff.difference_type == DIFF_MISSING_CANDIDATE_VALUE
    assert prof_diff.is_drift is False


def test_tag_dictionaries_missing_on_both_sides():
    """When tags are empty or None on both sides, they are categorized as UNCOMPARABLE with is_drift=False."""
    ref = create_sample_result(container_tags={})
    ref.video_streams[0].raw_tags = {}
    ref.audio_streams[0].raw_tags = {}

    cand = create_sample_result(container_tags={})
    cand.video_streams[0].raw_tags = {}
    cand.audio_streams[0].raw_tags = {}

    res = compare_technical_metadata(ref, cand)

    assert res.available is True
    assert res.status == STATUS_SUCCESS_WITH_MISSING_FIELDS
    assert res.summary.changed_fields_count == 0
    assert res.summary.has_transcoding_drift is False

    container_tag_diff = next(d for d in res.differences if d.field_path == "container.tags")
    assert container_tag_diff.difference_type == DIFF_UNCOMPARABLE
    assert container_tag_diff.comparison_outcome == OUTCOME_UNCOMPARABLE
    assert container_tag_diff.is_drift is False
    assert res.summary.uncomparable_fields_count >= 3
    assert res.summary.jointly_unavailable_fields_count >= 3

