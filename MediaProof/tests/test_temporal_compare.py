"""
Tests for Temporal dHash Sequence Comparison (Phase 1.2).

Covers all 14 required validation scenarios:
1. Hamming distance of identical hashes equals zero.
2. Hamming distance of known bit patterns is correct, including all-zeros vs all-ones equals 64.
3. Normalized Hamming distance is correct.
4. Identical sequences match deterministically.
5. Small timestamp offsets are handled according to the documented matching strategy.
6. Extra candidate samples are reported without corrupting alignment.
7. Missing candidate samples are reported.
8. A shorter candidate segment is handled and reported.
9. Dissimilar sequences produce low similarity evidence without a manipulation verdict.
10. Unavailable extraction results are rejected safely.
11. Empty sequences are rejected safely.
12. Malformed hashes and invalid timestamp ordering are handled explicitly.
13. Repeated comparison of the same inputs produces the same result.
14. The configured length/runtime guard is tested.
"""

import math
from pathlib import Path
import pytest

from forensics.temporal_dhash import (
    TemporalDHashResult,
    TemporalDHashSample,
)
from forensics.temporal_compare import (
    DEFAULT_MAX_HAMMING_THRESHOLD,
    STATUS_EMPTY_SEQUENCE,
    STATUS_INVALID_INPUT_TYPE,
    STATUS_INVALID_TIMESTAMP_ORDERING,
    STATUS_MALFORMED_HASH,
    STATUS_SEQUENCE_TOO_LONG,
    STATUS_SUCCESS,
    STATUS_UNAVAILABLE_INPUT,
    MatchedSamplePair,
    TemporalComparisonResult,
    compare_temporal_dhash_sequences,
    compute_hamming_distance,
    compute_normalized_hamming_distance,
)


# Helper to build synthetic sample sequences
def make_sample(
    index: int,
    timestamp: float,
    dhash_int: int,
    frame_index: int = None,
) -> TemporalDHashSample:
    return TemporalDHashSample(
        sample_index=index,
        timestamp_seconds=timestamp,
        dhash_hex=f"{dhash_int:016x}",
        dhash_int=dhash_int,
        frame_index=frame_index if frame_index is not None else index * 10,
    )


def make_result(
    samples: list,
    available: bool = True,
    status: str = "SUCCESS",
) -> TemporalDHashResult:
    return TemporalDHashResult(
        available=available,
        status=status,
        samples=samples,
        sample_count=len(samples),
        sampling_interval_seconds=1.0,
    )


# ==============================================================================
# Test 1: Hamming distance of identical hashes equals zero
# ==============================================================================

def test_hamming_distance_identical_hashes():
    # Ints
    assert compute_hamming_distance(0, 0) == 0
    assert compute_hamming_distance(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF) == 0
    assert compute_hamming_distance(0x123456789ABCDEF0, 0x123456789ABCDEF0) == 0

    # Hex strings
    assert compute_hamming_distance("0000000000000000", "0000000000000000") == 0
    assert compute_hamming_distance("ffffffffffffffff", "ffffffffffffffff") == 0
    assert compute_hamming_distance("123456789abcdef0", "123456789abcdef0") == 0

    # Cross-type (int vs hex string)
    assert compute_hamming_distance(0x123456789ABCDEF0, "123456789abcdef0") == 0


# ==============================================================================
# Test 2: Hamming distance of known bit patterns
# ==============================================================================

def test_hamming_distance_known_bit_patterns():
    # All zeros vs all ones = 64
    assert compute_hamming_distance(0, 0xFFFFFFFFFFFFFFFF) == 64
    assert compute_hamming_distance("0000000000000000", "ffffffffffffffff") == 64

    # Single bit difference
    assert compute_hamming_distance(0, 1) == 1
    assert compute_hamming_distance(0x8000000000000000, 0) == 1
    assert compute_hamming_distance(0x0000000000000001, 0x0000000000000003) == 1

    # Alternating nibbles: 0xAAAAAAAAAAAAAAAA vs 0x5555555555555555 = 64
    assert compute_hamming_distance(0xAAAAAAAAAAAAAAAA, 0x5555555555555555) == 64

    # 4 bits differing
    assert compute_hamming_distance(0x000000000000000F, 0x0000000000000000) == 4


# ==============================================================================
# Test 3: Normalized Hamming distance is correct
# ==============================================================================

def test_normalized_hamming_distance():
    # Exactly 0.0 for identical
    assert compute_normalized_hamming_distance(0, 0) == 0.0
    assert compute_normalized_hamming_distance("ffffffffffffffff", "ffffffffffffffff") == 0.0

    # Exactly 1.0 for completely inverted
    assert compute_normalized_hamming_distance(0, 0xFFFFFFFFFFFFFFFF) == 1.0

    # Exactly 8 bits differing out of 64 = 0.125
    assert compute_normalized_hamming_distance(0, 0x00000000000000FF) == pytest.approx(8 / 64.0)

    # Exactly 16 bits differing out of 64 = 0.25
    assert compute_normalized_hamming_distance(0, 0x000000000000FFFF) == pytest.approx(16 / 64.0)


# ==============================================================================
# Test 4: Identical sequences match deterministically
# ==============================================================================

def test_identical_sequences_match_deterministically():
    hashes = [0x1000, 0x2000, 0x3000, 0x4000, 0x5000]
    ref_samples = [make_sample(i, float(i), h) for i, h in enumerate(hashes)]
    cand_samples = [make_sample(i, float(i), h) for i, h in enumerate(hashes)]

    ref_res = make_result(ref_samples)
    cand_res = make_result(cand_samples)

    result = compare_temporal_dhash_sequences(ref_res, cand_res)

    assert result.available is True
    assert result.status == STATUS_SUCCESS
    assert result.reference_sample_count == 5
    assert result.candidate_sample_count == 5
    assert result.matched_pairs_count == 5
    assert result.unmatched_reference_count == 0
    assert result.unmatched_candidate_count == 0
    assert result.match_ratio == 1.0
    assert result.reference_coverage == 1.0
    assert result.candidate_coverage == 1.0
    assert result.mean_hamming_distance == 0.0
    assert result.median_hamming_distance == 0.0
    assert result.max_hamming_distance == 0
    assert result.mean_timestamp_delta_seconds == 0.0

    # Verify matched pair alignment
    for i, pair in enumerate(result.matched_pairs):
        assert pair.reference_index == i
        assert pair.candidate_index == i
        assert pair.reference_timestamp_seconds == float(i)
        assert pair.candidate_timestamp_seconds == float(i)
        assert pair.hamming_distance == 0
        assert pair.normalized_hamming_distance == 0.0
        assert pair.timestamp_delta_seconds == 0.0


# ==============================================================================
# Test 5: Small timestamp offsets are handled according to matching strategy
# ==============================================================================

def test_small_timestamp_offsets_handled():
    hashes = [0x1111, 0x2222, 0x3333, 0x4444]
    # Reference at t = [0.0, 1.0, 2.0, 3.0]
    ref_samples = [make_sample(i, float(i), h) for i, h in enumerate(hashes)]
    # Candidate offset by +0.3 seconds: t = [0.3, 1.3, 2.3, 3.3]
    cand_samples = [make_sample(i, float(i) + 0.3, h) for i, h in enumerate(hashes)]

    result = compare_temporal_dhash_sequences(ref_samples, cand_samples)

    assert result.available is True
    assert result.status == STATUS_SUCCESS
    assert result.matched_pairs_count == 4
    assert result.unmatched_reference_count == 0
    assert result.unmatched_candidate_count == 0
    assert result.mean_hamming_distance == 0.0
    assert result.mean_timestamp_delta_seconds == pytest.approx(0.3)

    for i, pair in enumerate(result.matched_pairs):
        assert pair.reference_index == i
        assert pair.candidate_index == i
        assert pair.timestamp_delta_seconds == pytest.approx(0.3)


# ==============================================================================
# Test 6: Extra candidate samples reported without corrupting alignment
# ==============================================================================

def test_extra_candidate_samples_reported():
    # Reference: 4 frames
    ref_hashes = [0x1000, 0x2000, 0x3000, 0x4000]
    ref_samples = [make_sample(i, float(i), h) for i, h in enumerate(ref_hashes)]

    # Candidate has extra inserted frames at index 1 and index 4 with distinct hashes
    cand_samples = [
        make_sample(0, 0.0, 0x1000),          # matches ref 0
        make_sample(1, 0.5, 0xFAFAFAFAFAFA),  # extra inserted frame
        make_sample(2, 1.0, 0x2000),          # matches ref 1
        make_sample(3, 2.0, 0x3000),          # matches ref 2
        make_sample(4, 2.5, 0xEEEEEEEEEEEE),  # extra inserted frame
        make_sample(5, 3.0, 0x4000),          # matches ref 3
    ]

    result = compare_temporal_dhash_sequences(ref_samples, cand_samples)

    assert result.available is True
    assert result.status == STATUS_SUCCESS
    assert result.reference_sample_count == 4
    assert result.candidate_sample_count == 6
    assert result.matched_pairs_count == 4
    assert result.unmatched_reference_count == 0
    assert result.unmatched_candidate_count == 2
    assert result.unmatched_candidate_indices == [1, 4]
    assert result.unmatched_candidate_ranges == [[1, 1], [4, 4]]
    assert result.reference_coverage == 1.0
    assert result.candidate_coverage == pytest.approx(4 / 6.0)

    # Check alignment pairs
    expected_pairs = [(0, 0), (1, 2), (2, 3), (3, 5)]
    for (exp_r, exp_c), pair in zip(expected_pairs, result.matched_pairs):
        assert pair.reference_index == exp_r
        assert pair.candidate_index == exp_c


# ==============================================================================
# Test 7: Missing candidate samples are reported
# ==============================================================================

def test_missing_candidate_samples_reported():
    # Reference: 5 frames
    ref_hashes = [0x1000, 0x2000, 0x3000, 0x4000, 0x5000]
    ref_samples = [make_sample(i, float(i), h) for i, h in enumerate(ref_hashes)]

    # Candidate missing sample at ref index 2 (0x3000)
    cand_samples = [
        make_sample(0, 0.0, 0x1000),
        make_sample(1, 1.0, 0x2000),
        make_sample(2, 3.0, 0x4000),
        make_sample(3, 4.0, 0x5000),
    ]

    result = compare_temporal_dhash_sequences(ref_samples, cand_samples)

    assert result.available is True
    assert result.status == STATUS_SUCCESS
    assert result.reference_sample_count == 5
    assert result.candidate_sample_count == 4
    assert result.matched_pairs_count == 4
    assert result.unmatched_reference_count == 1
    assert result.unmatched_reference_indices == [2]
    assert result.unmatched_reference_ranges == [[2, 2]]
    assert result.unmatched_candidate_count == 0
    assert result.unmatched_candidate_indices == []
    assert result.reference_coverage == pytest.approx(4 / 5.0)
    assert result.candidate_coverage == 1.0

    expected_pairs = [(0, 0), (1, 1), (3, 2), (4, 3)]
    for (exp_r, exp_c), pair in zip(expected_pairs, result.matched_pairs):
        assert pair.reference_index == exp_r
        assert pair.candidate_index == exp_c


# ==============================================================================
# Test 8: A shorter candidate segment is handled and reported (subclip)
# ==============================================================================

def test_shorter_candidate_segment_handled():
    # Reference: 10 frames
    ref_hashes = [0x100 * (i + 1) for i in range(10)]
    ref_samples = [make_sample(i, float(i), h) for i, h in enumerate(ref_hashes)]

    # Candidate: Subclip matching ref indices 4 to 7 (4 frames)
    cand_samples = [
        make_sample(0, 0.0, ref_hashes[4]),
        make_sample(1, 1.0, ref_hashes[5]),
        make_sample(2, 2.0, ref_hashes[6]),
        make_sample(3, 3.0, ref_hashes[7]),
    ]

    result = compare_temporal_dhash_sequences(ref_samples, cand_samples)

    assert result.available is True
    assert result.status == STATUS_SUCCESS
    assert result.reference_sample_count == 10
    assert result.candidate_sample_count == 4
    assert result.matched_pairs_count == 4
    assert result.unmatched_candidate_count == 0
    assert result.unmatched_reference_count == 6
    assert result.unmatched_reference_indices == [0, 1, 2, 3, 8, 9]
    assert result.unmatched_reference_ranges == [[0, 3], [8, 9]]
    assert result.reference_coverage == pytest.approx(4 / 10.0)
    assert result.candidate_coverage == 1.0

    expected_pairs = [(4, 0), (5, 1), (6, 2), (7, 3)]
    for (exp_r, exp_c), pair in zip(expected_pairs, result.matched_pairs):
        assert pair.reference_index == exp_r
        assert pair.candidate_index == exp_c


# ==============================================================================
# Test 9: Dissimilar sequences produce low similarity evidence without verdict
# ==============================================================================

def test_dissimilar_sequences_produce_low_similarity():
    # All zeros vs all ones (64 bits different)
    ref_samples = [
        make_sample(0, 0.0, 0x0000000000000000),
        make_sample(1, 1.0, 0x0000000000000001),
        make_sample(2, 2.0, 0x0000000000000002),
    ]
    cand_samples = [
        make_sample(0, 0.0, 0xFFFFFFFFFFFFFFFF),
        make_sample(1, 1.0, 0xFFFFFFFFFFFFFFFE),
        make_sample(2, 2.0, 0xFFFFFFFFFFFFFFFD),
    ]

    result = compare_temporal_dhash_sequences(ref_samples, cand_samples)

    assert result.available is True
    assert result.status == STATUS_SUCCESS
    assert result.matched_pairs_count == 0
    assert result.unmatched_reference_count == 3
    assert result.unmatched_candidate_count == 3
    assert result.match_ratio == 0.0
    assert result.reference_coverage == 0.0
    assert result.candidate_coverage == 0.0
    assert result.mean_hamming_distance is None
    assert result.median_hamming_distance is None
    assert result.max_hamming_distance is None

    # Crucial architectural assertion: Must not output final classification verdicts
    result_dict = result.to_dict()
    forbidden_verdicts = [
        "Verified Original",
        "Verified Derivative",
        "Potentially Manipulated",
        "Unknown Provenance",
        "Insufficient Evidence",
    ]
    result_str = str(result_dict)
    for verdict in forbidden_verdicts:
        assert verdict not in result_str


# ==============================================================================
# Test 10: Unavailable extraction results are rejected safely
# ==============================================================================

def test_unavailable_extraction_rejected_safely():
    valid_samples = [make_sample(0, 0.0, 0x1234)]
    unavailable_res = TemporalDHashResult(
        available=False,
        status="UNAVAILABLE_DECODER_MISSING",
        samples=[],
    )

    # Reference unavailable
    res1 = compare_temporal_dhash_sequences(unavailable_res, valid_samples)
    assert res1.available is False
    assert res1.status == STATUS_UNAVAILABLE_INPUT
    assert "unavailable" in res1.details["error"]

    # Candidate unavailable
    res2 = compare_temporal_dhash_sequences(valid_samples, unavailable_res)
    assert res2.available is False
    assert res2.status == STATUS_UNAVAILABLE_INPUT
    assert "unavailable" in res2.details["error"]


# ==============================================================================
# Test 11: Empty sequences are rejected safely
# ==============================================================================

def test_empty_sequences_rejected_safely():
    valid_samples = [make_sample(0, 0.0, 0x1234)]
    empty_samples = []

    # Empty reference
    res1 = compare_temporal_dhash_sequences(empty_samples, valid_samples)
    assert res1.available is False
    assert res1.status == STATUS_EMPTY_SEQUENCE

    # Empty candidate
    res2 = compare_temporal_dhash_sequences(valid_samples, empty_samples)
    assert res2.available is False
    assert res2.status == STATUS_EMPTY_SEQUENCE


# ==============================================================================
# Test 12: Malformed hashes and invalid timestamp ordering handled explicitly
# ==============================================================================

def test_malformed_hashes_and_timestamp_ordering():
    # 1. Malformed hash: not a valid 64-bit int or 16-char hex
    bad_hash_samples = [
        {"sample_index": 0, "timestamp_seconds": 0.0, "dhash_hex": "not_hex_at_all"},
    ]
    valid_samples = [make_sample(0, 0.0, 0x1234)]
    res_bad_hash = compare_temporal_dhash_sequences(bad_hash_samples, valid_samples)
    assert res_bad_hash.available is False
    assert res_bad_hash.status == STATUS_MALFORMED_HASH

    # Hash out of 64-bit range
    overflow_hash_samples = [
        {"sample_index": 0, "timestamp_seconds": 0.0, "dhash_int": 1 << 65},
    ]
    res_overflow = compare_temporal_dhash_sequences(overflow_hash_samples, valid_samples)
    assert res_overflow.available is False
    assert res_overflow.status == STATUS_MALFORMED_HASH

    # 2. Non-increasing timestamps (must not silently sort or repair)
    out_of_order_samples = [
        make_sample(0, 1.0, 0x1000),
        make_sample(1, 0.5, 0x2000),  # violates monotonicity
    ]
    res_order = compare_temporal_dhash_sequences(out_of_order_samples, valid_samples)
    assert res_order.available is False
    assert res_order.status == STATUS_INVALID_TIMESTAMP_ORDERING
    assert res_order.details["violation_index"] == 1

    # Equal timestamps (duplicate timestamp)
    duplicate_ts_samples = [
        make_sample(0, 1.0, 0x1000),
        make_sample(1, 1.0, 0x2000),  # duplicate timestamp
    ]
    res_dup = compare_temporal_dhash_sequences(duplicate_ts_samples, valid_samples)
    assert res_dup.available is False
    assert res_dup.status == STATUS_INVALID_TIMESTAMP_ORDERING

    # Negative timestamp
    neg_ts_samples = [
        make_sample(0, -0.5, 0x1000),
    ]
    res_neg = compare_temporal_dhash_sequences(neg_ts_samples, valid_samples)
    assert res_neg.available is False
    assert res_neg.status == STATUS_INVALID_TIMESTAMP_ORDERING

    # NaN / Inf timestamp
    nan_ts_samples = [
        make_sample(0, float("nan"), 0x1000),
    ]
    res_nan = compare_temporal_dhash_sequences(nan_ts_samples, valid_samples)
    assert res_nan.available is False
    assert res_nan.status == STATUS_INVALID_TIMESTAMP_ORDERING


# ==============================================================================
# Test 13: Repeated comparison of the same inputs produces the same result
# ==============================================================================

def test_repeated_comparison_repeatability():
    ref_samples = [make_sample(i, float(i), 0x1000 * (i + 1)) for i in range(6)]
    cand_samples = [make_sample(i, float(i), 0x1000 * (i + 1)) for i in range(6)]

    res1 = compare_temporal_dhash_sequences(ref_samples, cand_samples)
    res2 = compare_temporal_dhash_sequences(ref_samples, cand_samples)

    d1 = res1.to_dict()
    d2 = res2.to_dict()
    d1.pop("processing_time_seconds", None)
    d2.pop("processing_time_seconds", None)

    assert d1 == d2
    assert res1.matched_pairs_count == res2.matched_pairs_count
    assert res1.mean_hamming_distance == res2.mean_hamming_distance
    assert res1.match_ratio == res2.match_ratio


# ==============================================================================
# Test 14: Configured length/runtime guard is tested
# ==============================================================================

def test_configured_length_runtime_guard():
    # Limit of 10 samples
    guard_limit = 10
    ref_samples = [make_sample(i, float(i), 0x1000) for i in range(15)]
    cand_samples = [make_sample(i, float(i), 0x1000) for i in range(5)]

    # Reference exceeds guard
    res_ref_exceed = compare_temporal_dhash_sequences(
        ref_samples,
        cand_samples,
        max_sequence_length=guard_limit,
    )
    assert res_ref_exceed.available is False
    assert res_ref_exceed.status == STATUS_SEQUENCE_TOO_LONG
    assert "exceeds maximum allowed limit" in res_ref_exceed.details["error"]

    # Candidate exceeds guard
    res_cand_exceed = compare_temporal_dhash_sequences(
        cand_samples,
        ref_samples,
        max_sequence_length=guard_limit,
    )
    assert res_cand_exceed.available is False
    assert res_cand_exceed.status == STATUS_SEQUENCE_TOO_LONG

    # Within guard succeeds
    res_ok = compare_temporal_dhash_sequences(
        cand_samples,
        cand_samples,
        max_sequence_length=guard_limit,
    )
    assert res_ok.available is True
    assert res_ok.status == STATUS_SUCCESS
