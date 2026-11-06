"""
Deterministic Temporal Perceptual Difference-Hash (dHash) Sequence Comparison.

Implements Layer 1 (Temporal Sequence Comparison) for MediaProof.
Compares two temporal fingerprint sequences (reference and candidate) extracted
via `forensics.temporal_dhash` to evaluate temporal visual continuity, edits,
subclip extraction, inserted/dropped frames, and perceptual drift.

Algorithm Specification:
1. Input Normalization & Strict Validation:
   - Accepts either `TemporalDHashResult` instances or validated sample sequences.
   - Enforces availability check: Extraction failures cannot be treated as clean.
   - Enforces non-empty check: Sequences must contain at least one sample.
   - Enforces strict hash validation: Hashes must be 64-bit unsigned integers or
     16-character hexadecimal strings.
   - Enforces strict timestamp monotonicity: Timestamps must strictly increase
     (t[i+1] > t[i]); malformed or out-of-order sequences are explicitly rejected
     rather than silently sorted or repaired.
   - Enforces sequence length guard: Sequences exceeding `max_sequence_length` are
     rejected to prevent quadratic runtime/memory exhaustion.

2. Bitwise Hamming Distance:
   - Distance between two 64-bit hashes is calculated as:
       distance = (hash_a ^ hash_b).bit_count()
   - Range is strictly [0, 64].
   - Normalized distance: distance / 64.0 (range [0.0, 1.0]).
   - A distance of 0 indicates bit-identical hashes; larger distances indicate
     increasing visual variation. Individual distance values alone do not prove
     tampering or manipulation.

3. Monotonic Dynamic Programming Sequence Alignment:
   - Time complexity: O(M * N), where M = len(reference), N = len(candidate).
   - Space complexity: O(M * N) for the DP cost and backtrack matrices, safely
     bounded by `max_sequence_length`.
   - Semi-global boundary conditions: Free leading and trailing gaps allow
     robust detection of subclips, trim edits, and time-shifted candidates
     without penalizing non-overlapping reference boundaries.
   - Internal gap penalty: Configurable penalty (default 1.0) applied when
     skipping samples within candidate or reference sequences, handling
     inserted frames (candidate extra) or dropped frames (missing candidate).
   - Match threshold: Two samples (r, c) may align if Hamming distance <= max_hamming_threshold.

4. Non-Verbal Forensic Evidence:
   - Produces structured evidence metrics (counts, coverage, mean/median distances,
     unmatched index ranges).
   - Does NOT issue authenticity or manipulation classifications (e.g. "Verified Original").
"""

from dataclasses import dataclass, field
import math
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from forensics.temporal_dhash import (
    DHASH_BITS,
    TemporalDHashResult,
    TemporalDHashSample,
)

ALGORITHM_NAME = "temporal_dhash_sequence_compare"
ALGORITHM_VERSION = "1.0.0"

# Configurable defaults
DEFAULT_MAX_HAMMING_THRESHOLD: int = 12  # <= 12 bits difference (>= 81.25% bit similarity)
DEFAULT_GAP_PENALTY: float = 1.0
DEFAULT_MAX_SEQUENCE_LENGTH: int = 7200  # 2 hours at 1.0s sampling interval

# Status constants
STATUS_SUCCESS = "SUCCESS"
STATUS_UNAVAILABLE_INPUT = "UNAVAILABLE_INPUT"
STATUS_EMPTY_SEQUENCE = "EMPTY_SEQUENCE"
STATUS_MALFORMED_HASH = "MALFORMED_HASH"
STATUS_INVALID_TIMESTAMP_ORDERING = "INVALID_TIMESTAMP_ORDERING"
STATUS_SEQUENCE_TOO_LONG = "SEQUENCE_TOO_LONG"
STATUS_INVALID_INPUT_TYPE = "INVALID_INPUT_TYPE"


@dataclass
class MatchedSamplePair:
    """An aligned pair of reference and candidate temporal dHash samples."""
    reference_index: int
    candidate_index: int
    reference_timestamp_seconds: float
    candidate_timestamp_seconds: float
    hamming_distance: int
    normalized_hamming_distance: float
    timestamp_delta_seconds: float  # candidate_timestamp - reference_timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_index": self.reference_index,
            "candidate_index": self.candidate_index,
            "reference_timestamp_seconds": round(self.reference_timestamp_seconds, 4),
            "candidate_timestamp_seconds": round(self.candidate_timestamp_seconds, 4),
            "hamming_distance": self.hamming_distance,
            "normalized_hamming_distance": round(self.normalized_hamming_distance, 4),
            "timestamp_delta_seconds": round(self.timestamp_delta_seconds, 4),
        }


@dataclass
class TemporalComparisonResult:
    """Structured result of temporal perceptual fingerprint sequence comparison."""
    available: bool
    status: str
    algorithm: str = f"{ALGORITHM_NAME}_v{ALGORITHM_VERSION}"
    reference_sample_count: int = 0
    candidate_sample_count: int = 0
    matched_pairs_count: int = 0
    unmatched_reference_count: int = 0
    unmatched_candidate_count: int = 0
    match_ratio: float = 0.0          # matched_pairs / max(ref_count, cand_count)
    reference_coverage: float = 0.0   # matched_pairs / ref_count
    candidate_coverage: float = 0.0   # matched_pairs / cand_count
    mean_hamming_distance: Optional[float] = None
    median_hamming_distance: Optional[float] = None
    max_hamming_distance: Optional[int] = None
    mean_timestamp_delta_seconds: Optional[float] = None
    matched_pairs: List[MatchedSamplePair] = field(default_factory=list)
    unmatched_reference_indices: List[int] = field(default_factory=list)
    unmatched_candidate_indices: List[int] = field(default_factory=list)
    unmatched_reference_ranges: List[List[int]] = field(default_factory=list)
    unmatched_candidate_ranges: List[List[int]] = field(default_factory=list)
    max_hamming_threshold: int = DEFAULT_MAX_HAMMING_THRESHOLD
    gap_penalty: float = DEFAULT_GAP_PENALTY
    processing_time_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "status": self.status,
            "algorithm": self.algorithm,
            "reference_sample_count": self.reference_sample_count,
            "candidate_sample_count": self.candidate_sample_count,
            "matched_pairs_count": self.matched_pairs_count,
            "unmatched_reference_count": self.unmatched_reference_count,
            "unmatched_candidate_count": self.unmatched_candidate_count,
            "match_ratio": round(self.match_ratio, 4),
            "reference_coverage": round(self.reference_coverage, 4),
            "candidate_coverage": round(self.candidate_coverage, 4),
            "mean_hamming_distance": (
                round(self.mean_hamming_distance, 4)
                if self.mean_hamming_distance is not None
                else None
            ),
            "median_hamming_distance": (
                round(self.median_hamming_distance, 4)
                if self.median_hamming_distance is not None
                else None
            ),
            "max_hamming_distance": self.max_hamming_distance,
            "mean_timestamp_delta_seconds": (
                round(self.mean_timestamp_delta_seconds, 4)
                if self.mean_timestamp_delta_seconds is not None
                else None
            ),
            "matched_pairs": [p.to_dict() for p in self.matched_pairs],
            "unmatched_reference_indices": self.unmatched_reference_indices,
            "unmatched_candidate_indices": self.unmatched_candidate_indices,
            "unmatched_reference_ranges": self.unmatched_reference_ranges,
            "unmatched_candidate_ranges": self.unmatched_candidate_ranges,
            "max_hamming_threshold": self.max_hamming_threshold,
            "gap_penalty": self.gap_penalty,
            "processing_time_seconds": round(self.processing_time_seconds, 4),
            "details": self.details,
        }


def compute_hamming_distance(hash_a: Union[int, str], hash_b: Union[int, str]) -> int:
    """
    Computes bitwise Hamming distance between two 64-bit dHashes.

    Args:
        hash_a: 64-bit integer or 16-character hex string.
        hash_b: 64-bit integer or 16-character hex string.

    Returns:
        Integer distance between 0 and 64 inclusive.

    Raises:
        ValueError: If either hash is not a valid 64-bit integer or 16-character hex string.
    """
    int_a = _parse_dhash_int(hash_a)
    int_b = _parse_dhash_int(hash_b)
    return (int_a ^ int_b).bit_count()


def compute_normalized_hamming_distance(hash_a: Union[int, str], hash_b: Union[int, str]) -> float:
    """
    Computes normalized Hamming distance (distance / 64.0) between two 64-bit dHashes.

    Returns:
        Float value between 0.0 (identical) and 1.0 (completely inverted).
    """
    return compute_hamming_distance(hash_a, hash_b) / float(DHASH_BITS)


def _parse_dhash_int(val: Union[int, str]) -> int:
    """Parses a hash representation into a 64-bit unsigned integer."""
    if isinstance(val, bool):
        raise ValueError("Boolean value is not a valid hash.")
    if isinstance(val, int):
        if 0 <= val < (1 << DHASH_BITS):
            return val
        raise ValueError(f"Hash integer out of range [0, 2^{DHASH_BITS}-1]: {val}")
    if isinstance(val, str):
        cleaned = val.strip()
        if len(cleaned) != 16:
            raise ValueError(f"Hex hash string must be exactly 16 characters: '{val}' (len={len(cleaned)})")
        try:
            parsed = int(cleaned, 16)
            if 0 <= parsed < (1 << DHASH_BITS):
                return parsed
        except ValueError as err:
            raise ValueError(f"Malformed hexadecimal hash string: '{val}'") from err
    raise ValueError(f"Unsupported hash type: {type(val).__name__}")


def _indices_to_ranges(indices: List[int]) -> List[List[int]]:
    """Converts a sorted list of discrete indices to contiguous [start, end] ranges."""
    if not indices:
        return []
    ranges: List[List[int]] = []
    start = indices[0]
    end = indices[0]
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            ranges.append([start, end])
            start = idx
            end = idx
    ranges.append([start, end])
    return ranges


def _validate_and_extract_samples(
    input_obj: Any,
    role_name: str,
    max_sequence_length: int,
) -> Tuple[bool, str, List[TemporalDHashSample], Dict[str, Any]]:
    """
    Validates an input extraction result or sample sequence.

    Args:
        input_obj: `TemporalDHashResult` or sequence of `TemporalDHashSample` or dicts.
        role_name: 'reference' or 'candidate' for clear error reporting.
        max_sequence_length: Maximum allowed number of samples.

    Returns:
        Tuple of (is_valid, status_code, validated_samples, error_details).
    """
    # 1. Check if input is TemporalDHashResult
    if isinstance(input_obj, TemporalDHashResult):
        if not input_obj.available:
            return (
                False,
                STATUS_UNAVAILABLE_INPUT,
                [],
                {"error": f"{role_name} extraction result marked unavailable ({input_obj.status})"},
            )
        raw_samples = input_obj.samples
    elif isinstance(input_obj, (list, tuple)):
        raw_samples = list(input_obj)
    else:
        return (
            False,
            STATUS_INVALID_INPUT_TYPE,
            [],
            {"error": f"{role_name} input must be TemporalDHashResult or sequence of samples, got {type(input_obj).__name__}"},
        )

    # 2. Check empty sequence
    if not raw_samples:
        return (
            False,
            STATUS_EMPTY_SEQUENCE,
            [],
            {"error": f"{role_name} sequence contains 0 samples"},
        )

    # 3. Check sequence length guard
    if len(raw_samples) > max_sequence_length:
        return (
            False,
            STATUS_SEQUENCE_TOO_LONG,
            [],
            {
                "error": f"{role_name} sequence length ({len(raw_samples)}) exceeds maximum allowed limit ({max_sequence_length})",
                "sample_count": len(raw_samples),
                "max_sequence_length": max_sequence_length,
            },
        )

    # 4. Validate each sample, hash format, and strictly increasing timestamps
    validated: List[TemporalDHashSample] = []
    last_timestamp: Optional[float] = None

    for i, item in enumerate(raw_samples):
        if isinstance(item, TemporalDHashSample):
            sample_index = item.sample_index
            timestamp = item.timestamp_seconds
            frame_index = item.frame_index
            raw_hash = item.dhash_int if item.dhash_int is not None else item.dhash_hex
        elif isinstance(item, dict):
            sample_index = item.get("sample_index", i)
            timestamp = item.get("timestamp_seconds")
            frame_index = item.get("frame_index")
            raw_hash = item.get("dhash_int", item.get("dhash_hex"))
        else:
            return (
                False,
                STATUS_INVALID_INPUT_TYPE,
                [],
                {"error": f"{role_name} sample at index {i} has unsupported type {type(item).__name__}"},
            )

        # Validate timestamp
        if timestamp is None or not isinstance(timestamp, (int, float)) or math.isnan(timestamp) or math.isinf(timestamp):
            return (
                False,
                STATUS_INVALID_TIMESTAMP_ORDERING,
                [],
                {"error": f"{role_name} sample at index {i} has invalid timestamp: {timestamp}"},
            )
        timestamp = float(timestamp)
        if timestamp < 0.0:
            return (
                False,
                STATUS_INVALID_TIMESTAMP_ORDERING,
                [],
                {"error": f"{role_name} sample at index {i} has negative timestamp: {timestamp}"},
            )

        # Enforce strict timestamp monotonicity (t[i] > t[i-1])
        if last_timestamp is not None and timestamp <= last_timestamp:
            return (
                False,
                STATUS_INVALID_TIMESTAMP_ORDERING,
                [],
                {
                    "error": (
                        f"{role_name} sequence violates strict timestamp monotonicity: "
                        f"sample {i} (t={timestamp:.4f}s) <= sample {i-1} (t={last_timestamp:.4f}s)"
                    ),
                    "violation_index": i,
                    "previous_timestamp": last_timestamp,
                    "current_timestamp": timestamp,
                },
            )
        last_timestamp = timestamp

        # Validate hash
        try:
            dhash_int = _parse_dhash_int(raw_hash)
            dhash_hex = f"{dhash_int:016x}"
        except (ValueError, TypeError) as err:
            return (
                False,
                STATUS_MALFORMED_HASH,
                [],
                {
                    "error": f"{role_name} sample at index {i} has malformed hash: {err}",
                    "sample_index": i,
                },
            )

        validated.append(
            TemporalDHashSample(
                sample_index=sample_index,
                timestamp_seconds=timestamp,
                dhash_hex=dhash_hex,
                dhash_int=dhash_int,
                frame_index=frame_index,
            )
        )

    return True, STATUS_SUCCESS, validated, {}


def align_temporal_sequences(
    reference_samples: List[TemporalDHashSample],
    candidate_samples: List[TemporalDHashSample],
    max_hamming_threshold: int = DEFAULT_MAX_HAMMING_THRESHOLD,
    gap_penalty: float = DEFAULT_GAP_PENALTY,
) -> Tuple[List[MatchedSamplePair], List[int], List[int]]:
    """
    Performs monotonic dynamic programming alignment between reference and candidate samples.

    Strategy:
    - Employs semi-global alignment with free leading/trailing reference gaps, allowing
      candidate clips to start and end at arbitrary reference offsets (e.g. subclip detection).
    - Transitions allow:
      1. Diagonal match: reference[i] and candidate[j] match if Hamming distance <= threshold.
         Match gain rewards low Hamming distance: (threshold - distance + 1.0).
      2. Candidate gap (insertion): candidate sample skipped at cost of `gap_penalty`.
      3. Reference gap (deletion): reference sample skipped at cost of `gap_penalty`.
    - Enforces strict monotonic time progression:
      reference indices: i_0 < i_1 < ... < i_k
      candidate indices: j_0 < j_1 < ... < j_k

    Complexity:
      Time: O(M * N)
      Space: O(M * N)

    Returns:
        Tuple of:
          - matched_pairs: List[MatchedSamplePair]
          - unmatched_reference_indices: List[int]
          - unmatched_candidate_indices: List[int]
    """
    M = len(reference_samples)
    N = len(candidate_samples)

    if M == 0 or N == 0:
        return [], list(range(M)), list(range(N))

    # Pre-extract integer hashes and timestamps
    ref_hashes = [s.dhash_int for s in reference_samples]
    cand_hashes = [s.dhash_int for s in candidate_samples]
    ref_ts = [s.timestamp_seconds for s in reference_samples]
    cand_ts = [s.timestamp_seconds for s in candidate_samples]

    # DP tables:
    # dp[i][j]: optimal alignment score considering ref[:i] and cand[:j]
    # backtrack[i][j]: (prev_i, prev_j)
    dp = [[0.0] * (N + 1) for _ in range(M + 1)]
    backtrack = [[(0, 0)] * (N + 1) for _ in range(M + 1)]

    # Dynamic programming fill
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            # Option 1: Skip reference sample (gap in candidate / deletion in candidate)
            best_val = dp[i - 1][j] - gap_penalty
            best_ptr = (i - 1, j)

            # Option 2: Skip candidate sample (gap in reference / extra candidate insertion)
            cand_gap_val = dp[i][j - 1] - gap_penalty
            if cand_gap_val > best_val:
                best_val = cand_gap_val
                best_ptr = (i, j - 1)

            # Option 3: Match pair (reference[i-1], candidate[j-1])
            dist = (ref_hashes[i - 1] ^ cand_hashes[j - 1]).bit_count()
            if dist <= max_hamming_threshold:
                match_gain = float(max_hamming_threshold - dist + 1.0)
                # Maximize from previous match, or start fresh if free leading gap
                match_val = max(0.0, dp[i - 1][j - 1]) + match_gain
                if match_val > best_val:
                    best_val = match_val
                    best_ptr = (i - 1, j - 1)

            dp[i][j] = best_val
            backtrack[i][j] = best_ptr

    # Identify optimal alignment endpoint
    # We inspect all cells to allow arbitrary subclip termination with free trailing gaps
    best_score = 0.0
    best_end = (0, 0)
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if dp[i][j] > best_score:
                best_score = dp[i][j]
                best_end = (i, j)

    # Traceback optimal alignment path
    matched_pairs: List[MatchedSamplePair] = []
    matched_ref_indices = set()
    matched_cand_indices = set()

    curr = best_end
    while curr[0] > 0 and curr[1] > 0 and dp[curr[0]][curr[1]] > 0.0:
        prev = backtrack[curr[0]][curr[1]]
        if prev == (curr[0] - 1, curr[1] - 1):
            r_idx = curr[0] - 1
            c_idx = curr[1] - 1
            dist = (ref_hashes[r_idx] ^ cand_hashes[c_idx]).bit_count()
            if dist <= max_hamming_threshold:
                matched_pairs.append(
                    MatchedSamplePair(
                        reference_index=r_idx,
                        candidate_index=c_idx,
                        reference_timestamp_seconds=ref_ts[r_idx],
                        candidate_timestamp_seconds=cand_ts[c_idx],
                        hamming_distance=dist,
                        normalized_hamming_distance=dist / float(DHASH_BITS),
                        timestamp_delta_seconds=cand_ts[c_idx] - ref_ts[r_idx],
                    )
                )
                matched_ref_indices.add(r_idx)
                matched_cand_indices.add(c_idx)
        curr = prev

    # Reverse matched pairs so they are in chronological order
    matched_pairs.reverse()

    # Determine unmatched sample indices
    unmatched_ref = [i for i in range(M) if i not in matched_ref_indices]
    unmatched_cand = [j for j in range(N) if j not in matched_cand_indices]

    return matched_pairs, unmatched_ref, unmatched_cand


def compare_temporal_dhash_sequences(
    reference: Union[TemporalDHashResult, Sequence[Any]],
    candidate: Union[TemporalDHashResult, Sequence[Any]],
    max_hamming_threshold: int = DEFAULT_MAX_HAMMING_THRESHOLD,
    gap_penalty: float = DEFAULT_GAP_PENALTY,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
) -> TemporalComparisonResult:
    """
    Compares two temporal dHash sequences and produces structured comparison evidence.

    Args:
        reference: `TemporalDHashResult` or sequence of samples from registered video.
        candidate: `TemporalDHashResult` or sequence of samples from submitted video.
        max_hamming_threshold: Maximum bitwise Hamming distance to treat a frame pair as matched.
        gap_penalty: Alignment penalty for unmatched intermediate frames.
        max_sequence_length: Maximum allowed sample sequence length before rejection.

    Returns:
        `TemporalComparisonResult` containing detailed alignment evidence.
    """
    start_time = time.perf_counter()

    # 1. Validate reference sequence
    ref_ok, ref_status, ref_samples, ref_details = _validate_and_extract_samples(
        reference, "Reference", max_sequence_length
    )
    if not ref_ok:
        return TemporalComparisonResult(
            available=False,
            status=ref_status,
            max_hamming_threshold=max_hamming_threshold,
            gap_penalty=gap_penalty,
            processing_time_seconds=time.perf_counter() - start_time,
            details=ref_details,
        )

    # 2. Validate candidate sequence
    cand_ok, cand_status, cand_samples, cand_details = _validate_and_extract_samples(
        candidate, "Candidate", max_sequence_length
    )
    if not cand_ok:
        return TemporalComparisonResult(
            available=False,
            status=cand_status,
            reference_sample_count=len(ref_samples),
            max_hamming_threshold=max_hamming_threshold,
            gap_penalty=gap_penalty,
            processing_time_seconds=time.perf_counter() - start_time,
            details=cand_details,
        )

    M = len(ref_samples)
    N = len(cand_samples)

    # 3. Perform dynamic programming alignment
    matched_pairs, unmatched_ref, unmatched_cand = align_temporal_sequences(
        reference_samples=ref_samples,
        candidate_samples=cand_samples,
        max_hamming_threshold=max_hamming_threshold,
        gap_penalty=gap_penalty,
    )

    matched_count = len(matched_pairs)
    unmatched_ref_count = len(unmatched_ref)
    unmatched_cand_count = len(unmatched_cand)

    # 4. Compute statistical summaries
    max_count = max(M, N)
    match_ratio = matched_count / float(max_count) if max_count > 0 else 0.0
    ref_coverage = matched_count / float(M) if M > 0 else 0.0
    cand_coverage = matched_count / float(N) if N > 0 else 0.0

    if matched_pairs:
        distances = [p.hamming_distance for p in matched_pairs]
        deltas = [p.timestamp_delta_seconds for p in matched_pairs]
        mean_dist = float(statistics.mean(distances))
        median_dist = float(statistics.median(distances))
        max_dist = max(distances)
        mean_delta = float(statistics.mean(deltas))
    else:
        mean_dist = None
        median_dist = None
        max_dist = None
        mean_delta = None

    unmatched_ref_ranges = _indices_to_ranges(unmatched_ref)
    unmatched_cand_ranges = _indices_to_ranges(unmatched_cand)

    details: Dict[str, Any] = {
        "alignment_strategy": "monotonic_dp_semi_global",
        "hamming_bits": DHASH_BITS,
        "notes": (
            "Evidence only; Hamming distance reflects bitwise perceptual variation "
            "and does not constitute definitive proof of authenticity or manipulation."
        ),
    }

    elapsed = time.perf_counter() - start_time

    return TemporalComparisonResult(
        available=True,
        status=STATUS_SUCCESS,
        reference_sample_count=M,
        candidate_sample_count=N,
        matched_pairs_count=matched_count,
        unmatched_reference_count=unmatched_ref_count,
        unmatched_candidate_count=unmatched_cand_count,
        match_ratio=match_ratio,
        reference_coverage=ref_coverage,
        candidate_coverage=cand_coverage,
        mean_hamming_distance=mean_dist,
        median_hamming_distance=median_dist,
        max_hamming_distance=max_dist,
        mean_timestamp_delta_seconds=mean_delta,
        matched_pairs=matched_pairs,
        unmatched_reference_indices=unmatched_ref,
        unmatched_candidate_indices=unmatched_cand,
        unmatched_reference_ranges=unmatched_ref_ranges,
        unmatched_candidate_ranges=unmatched_cand_ranges,
        max_hamming_threshold=max_hamming_threshold,
        gap_penalty=gap_penalty,
        processing_time_seconds=elapsed,
        details=details,
    )
