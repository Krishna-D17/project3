"""
Technical Metadata Drift Comparison.

Implements Layer 2 (Technical Metadata Drift Comparison) for MediaProof.
Compares technical metadata extracted from a registered reference media asset
and a candidate media file to systematically identify, categorize, and explain
differences resulting from transcoding, re-encoding, resizing, stream changes,
metadata stripping, or other transformations.

Architecture & Contract:
- Comparison only: Emits factual, structured technical drift evidence.
  Does NOT assign final MediaProof verification classifications (e.g. "Verified Original",
  "Potentially Manipulated") or modify the decision engine.
- Non-destructive & deterministic: Does not modify either input object. Repeated
  comparisons of identical inputs produce bit-for-bit identical evidence payloads.
- Global stream alignment: Employs a deterministic, global maximum-weight bipartite
  matching strategy that enforces one-to-one correspondence, resolves greedy conflicts,
  and explicitly identifies and flags ambiguous correspondences.
- Explicit failure & nullability semantics: Distinguishes unavailable inputs,
  missing fields, jointly unavailable properties, and genuine differences without fabricating defaults.
- Reproducible counting: Summary counts strictly partition all compared field records:
  total_compared_fields == unchanged + changed + missing + uncomparable.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from forensics.metadata_drift import (
    AudioStreamMetadata,
    ContainerMetadata,
    TechnicalMetadataResult,
    VideoStreamMetadata,
)

ALGORITHM_NAME = "technical_metadata_drift_compare"
ALGORITHM_VERSION = "1.0.0"

# Overall Comparison Statuses
STATUS_SUCCESS = "SUCCESS"
STATUS_SUCCESS_WITH_DRIFT = "SUCCESS_WITH_DRIFT"
STATUS_SUCCESS_WITH_MISSING_FIELDS = "SUCCESS_WITH_MISSING_FIELDS"
STATUS_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
STATUS_INVALID_INPUT = "INVALID_INPUT"
STATUS_INVALID_ARGUMENTS = STATUS_INVALID_INPUT  # Backwards compatibility alias

# Difference Types
DIFF_UNCHANGED = "UNCHANGED"
DIFF_CHANGED = "CHANGED"
DIFF_ADDED = "ADDED"
DIFF_REMOVED = "REMOVED"
DIFF_MISSING_REFERENCE_VALUE = "MISSING_REFERENCE_VALUE"
DIFF_MISSING_CANDIDATE_VALUE = "MISSING_CANDIDATE_VALUE"
DIFF_UNCOMPARABLE = "UNCOMPARABLE"

# Comparison Outcomes
OUTCOME_MATCH = "MATCH"
OUTCOME_DRIFT = "DRIFT"
OUTCOME_MISSING = "MISSING"
OUTCOME_UNCOMPARABLE = "UNCOMPARABLE"

# Stream Matching Statuses
STREAM_MATCHED = "MATCHED"
STREAM_REMOVED = "REMOVED"
STREAM_ADDED = "ADDED"
STREAM_AMBIGUOUS = "AMBIGUOUS"

# Configurable Defaults
DEFAULT_DURATION_TOLERANCE_SECONDS: float = 0.05
DEFAULT_FPS_TOLERANCE: float = 0.005
MIN_PLAUSIBLE_AFFINITY: int = 1
DEFAULT_MAX_MATCHING_EVALUATIONS: int = 10000
DEFAULT_MAX_OPTIMAL_MAPPINGS: int = 100



# ==============================================================================
# Data Models
# ==============================================================================

@dataclass
class MetadataFieldDifference:
    """A single field-level comparison record between reference and candidate metadata."""
    field_path: str
    metadata_category: str  # "container", "video", "audio", "stream_count"
    reference_value: Any
    candidate_value: Any
    difference_type: str  # DIFF_UNCHANGED, DIFF_CHANGED, etc.
    comparison_outcome: str  # OUTCOME_MATCH, OUTCOME_DRIFT, OUTCOME_MISSING, OUTCOME_UNCOMPARABLE
    explanation: str
    is_drift: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_path": self.field_path,
            "metadata_category": self.metadata_category,
            "reference_value": self.reference_value,
            "candidate_value": self.candidate_value,
            "difference_type": self.difference_type,
            "comparison_outcome": self.comparison_outcome,
            "explanation": self.explanation,
            "is_drift": self.is_drift,
        }


@dataclass
class StreamMatchRecord:
    """Details of alignment between a reference stream and a candidate stream."""
    stream_type: str  # "video" or "audio"
    reference_stream_index: Optional[int]
    candidate_stream_index: Optional[int]
    match_status: str  # STREAM_MATCHED, STREAM_REMOVED, STREAM_ADDED, STREAM_AMBIGUOUS
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_type": self.stream_type,
            "reference_stream_index": self.reference_stream_index,
            "candidate_stream_index": self.candidate_stream_index,
            "match_status": self.match_status,
            "explanation": self.explanation,
        }


@dataclass
class MetadataComparisonSummary:
    """Reproducible summary counts and high-level drift descriptors."""
    total_compared_fields: int = 0
    unchanged_fields_count: int = 0
    changed_fields_count: int = 0
    missing_fields_count: int = 0
    uncomparable_fields_count: int = 0
    jointly_unavailable_fields_count: int = 0
    added_streams_count: int = 0
    removed_streams_count: int = 0
    has_critical_drift: bool = False  # e.g., resolution change, material duration discrepancy
    has_transcoding_drift: bool = False  # e.g., bitrate, codec, encoder tag

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_compared_fields": self.total_compared_fields,
            "unchanged_fields_count": self.unchanged_fields_count,
            "changed_fields_count": self.changed_fields_count,
            "missing_fields_count": self.missing_fields_count,
            "uncomparable_fields_count": self.uncomparable_fields_count,
            "jointly_unavailable_fields_count": self.jointly_unavailable_fields_count,
            "added_streams_count": self.added_streams_count,
            "removed_streams_count": self.removed_streams_count,
            "has_critical_drift": self.has_critical_drift,
            "has_transcoding_drift": self.has_transcoding_drift,
        }


@dataclass
class MetadataComparisonResult:
    """Structured result of technical metadata comparison between reference and candidate media."""
    available: bool
    status: str
    algorithm: str = f"{ALGORITHM_NAME}_v{ALGORITHM_VERSION}"
    differences: List[MetadataFieldDifference] = field(default_factory=list)
    summary: MetadataComparisonSummary = field(default_factory=MetadataComparisonSummary)
    stream_matches: List[StreamMatchRecord] = field(default_factory=list)
    reference_file_path: Optional[str] = None
    candidate_file_path: Optional[str] = None
    processing_time_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_runtime_measurements: bool = True) -> Dict[str, Any]:
        """
        Serializes comparison result to a structured dictionary.

        Args:
            include_runtime_measurements: If True, includes nondeterministic runtime
                measurements ('processing_time_seconds'). If False, emits strictly
                deterministic evidence payload.
        """
        payload: Dict[str, Any] = {
            "available": self.available,
            "status": self.status,
            "algorithm": self.algorithm,
            "differences": [d.to_dict() for d in self.differences],
            "summary": self.summary.to_dict(),
            "stream_matches": [m.to_dict() for m in self.stream_matches],
            "reference_file_path": self.reference_file_path,
            "candidate_file_path": self.candidate_file_path,
            "details": self.details,
        }
        if include_runtime_measurements:
            payload["processing_time_seconds"] = round(self.processing_time_seconds, 4)
        return payload

    def to_evidence_dict(self) -> Dict[str, Any]:
        """
        Returns strictly deterministic comparison evidence payload.
        Guaranteed to produce identical serialized output across repeated runs on identical inputs.
        Excludes nondeterministic operational timing.
        """
        return self.to_dict(include_runtime_measurements=False)


# ==============================================================================
# Documented Container Format Alias Families & Normalization Policy
# ==============================================================================
# FFprobe reports container formats via demuxer names, commonly producing compound
# comma-delimited strings representing supported format families
# (e.g., 'mov,mp4,m4a,3gp,3g2,mj2').
#
# Policy:
# 1. Exact string match (case-insensitive, whitespace-stripped) -> UNCHANGED / OUTCOME_MATCH (is_drift=False).
# 2. Documented Format Families: If both format identifiers belong to the same canonical
#    format family (e.g. 'mov,mp4,m4a,3gp,3g2,mj2' and 'mp4', or 'matroska,webm' and 'webm'),
#    they are recognized as compatible representations of the same container family
#    and categorized as UNCHANGED / OUTCOME_MATCH (is_drift=False).
# 3. Demuxer-Grounded Conservative Scope: Aliases must reflect valid representations of FFprobe's
#    format_name field. Distinct demuxer formats must NOT be classified as equivalent merely
#    because their names are associated with the same broader ecosystem:
#    - MP4 brand identifiers ('isom', 'mp41', 'mp42') are metadata tags, not container format aliases.
#    - Codec/elementary stream identifiers ('opus') are not container format aliases for 'ogg'.
# 4. Non-Equivalence of Arbitrary Shared Tokens: A shared token between arbitrary, unregistered
#    format strings does NOT prove equivalence. Arbitrary strings without an explicit family
#    mapping or exact match are reported as CHANGED / OUTCOME_DRIFT (is_drift=True).
# 5. Genuinely Different Formats: Genuinely distinct formats (e.g., 'mp4' vs 'matroska,webm'
#    or 'mp4' vs 'avi') are reported as CHANGED / OUTCOME_DRIFT (is_drift=True).
# 6. Missing Evidence Semantics: If format name is absent from one side, it is reported as
#    MISSING_REFERENCE_VALUE or MISSING_CANDIDATE_VALUE with is_drift=False.
# 7. Jointly Unavailable: If format name is absent on both sides, it is reported as
#    UNCOMPARABLE with is_drift=False.
#
# Limitations:
# - FFprobe format names reflect demuxer binding, not stream-level encoding.
# - Containers sharing an alias family may differ in minor brand tags; this rule
#   evaluates container family equivalence, while stream codecs and tags are checked
#   independently.

KNOWN_FORMAT_FAMILIES: Dict[str, Set[str]] = {
    "mp4_mov": {
        "mov", "mp4", "m4a", "3gp", "3g2", "mj2", "quicktime",
        "mov,mp4,m4a,3gp,3g2,mj2",
    },
    "matroska_webm": {
        "matroska", "webm", "mkv", "matroska,webm",
    },
    "ogg": {
        "ogg", "ogv", "oga",
    },
    "mpegts": {
        "mpegts", "ts", "m2ts",
    },
    "mpeg": {
        "mpeg", "vob", "mpg",
    },
    "avi": {
        "avi",
    },
    "flv": {
        "flv",
    },
    "wav": {
        "wav", "wave",
    },
    "mp3": {
        "mp3",
    },
    "flac": {
        "flac",
    },
    "aac": {
        "aac",
    },
}


def normalize_format_set(fmt_str: Optional[str]) -> Set[str]:
    """Splits format name aliases (e.g., 'mov,mp4,m4a,3gp,3g2,mj2') into normalized tokens."""
    if not fmt_str or not isinstance(fmt_str, str):
        return set()
    return {token.strip().lower() for token in fmt_str.split(",") if token.strip()}


def resolve_format_family(fmt_str: Optional[str]) -> Optional[str]:
    """
    Resolves an FFprobe format name string to a canonical format family.
    Returns the canonical family name if uniquely resolved, or None if unrecognized.
    """
    if not fmt_str or not isinstance(fmt_str, str):
        return None
    clean = fmt_str.strip().lower()
    if not clean:
        return None

    # Check whole string match
    for fam_name, members in KNOWN_FORMAT_FAMILIES.items():
        if clean in members:
            return fam_name

    # Check token subset match
    tokens = normalize_format_set(fmt_str)
    if not tokens:
        return None

    matching_families = []
    for fam_name, members in KNOWN_FORMAT_FAMILIES.items():
        if tokens.issubset(members):
            matching_families.append(fam_name)

    if len(matching_families) == 1:
        return matching_families[0]

    return None


def compare_format_names(ref_fmt: Optional[str], cand_fmt: Optional[str]) -> Tuple[str, str, str, bool]:
    """
    Compares container format names considering compatible aliases.
    Returns: (diff_type, outcome, explanation, is_drift)
    """
    if ref_fmt is None and cand_fmt is None:
        return (
            DIFF_UNCOMPARABLE,
            OUTCOME_UNCOMPARABLE,
            "Both container format names are unavailable; property cannot be compared or positively matched",
            False,
        )
    if ref_fmt is None:
        return (
            DIFF_MISSING_REFERENCE_VALUE,
            OUTCOME_MISSING,
            f"Reference container format name missing; candidate is '{cand_fmt}'",
            False,
        )
    if cand_fmt is None:
        return (
            DIFF_MISSING_CANDIDATE_VALUE,
            OUTCOME_MISSING,
            f"Candidate container format name missing; reference is '{ref_fmt}'",
            False,
        )

    clean_ref = ref_fmt.strip().lower()
    clean_cand = cand_fmt.strip().lower()

    if clean_ref == clean_cand:
        return DIFF_UNCHANGED, OUTCOME_MATCH, f"Container format matches exactly ('{ref_fmt}')", False

    # Check known format families
    ref_fam = resolve_format_family(ref_fmt)
    cand_fam = resolve_format_family(cand_fmt)

    if ref_fam is not None and cand_fam is not None and ref_fam == cand_fam:
        return (
            DIFF_UNCHANGED,
            OUTCOME_MATCH,
            f"Container format matches compatible known alias family '{ref_fam}' ('{ref_fmt}' vs '{cand_fmt}')",
            False,
        )

    return (
        DIFF_CHANGED,
        OUTCOME_DRIFT,
        f"Container format changed from '{ref_fmt}' to '{cand_fmt}'",
        True,
    )


def compare_scalar_field(
    field_path: str,
    category: str,
    ref_val: Any,
    cand_val: Any,
    numeric_tolerance: Optional[float] = None,
    field_unit: str = "",
) -> MetadataFieldDifference:
    """
    Systematically compares a scalar field between reference and candidate.
    Categorizes the difference into UNCHANGED, CHANGED, MISSING_REFERENCE_VALUE,
    MISSING_CANDIDATE_VALUE, or UNCOMPARABLE (when both absent).
    """
    unit_str = f" {field_unit}" if field_unit else ""

    # Both missing: Explicitly categorized as UNCOMPARABLE rather than falsely verified as equal
    if ref_val is None and cand_val is None:
        return MetadataFieldDifference(
            field_path=field_path,
            metadata_category=category,
            reference_value=None,
            candidate_value=None,
            difference_type=DIFF_UNCOMPARABLE,
            comparison_outcome=OUTCOME_UNCOMPARABLE,
            explanation=f"{field_path} is unavailable in both reference and candidate; property cannot be compared or positively matched",
            is_drift=False,
        )

    # Missing in reference
    if ref_val is None:
        return MetadataFieldDifference(
            field_path=field_path,
            metadata_category=category,
            reference_value=None,
            candidate_value=cand_val,
            difference_type=DIFF_MISSING_REFERENCE_VALUE,
            comparison_outcome=OUTCOME_MISSING,
            explanation=f"Missing in reference; candidate reports {cand_val}{unit_str}",
            is_drift=False,
        )

    # Missing in candidate
    if cand_val is None:
        return MetadataFieldDifference(
            field_path=field_path,
            metadata_category=category,
            reference_value=ref_val,
            candidate_value=None,
            difference_type=DIFF_MISSING_CANDIDATE_VALUE,
            comparison_outcome=OUTCOME_MISSING,
            explanation=f"Missing in candidate; reference reports {ref_val}{unit_str}",
            is_drift=False,
        )

    # Numeric comparison with tolerance
    if (
        numeric_tolerance is not None
        and isinstance(ref_val, (int, float))
        and isinstance(cand_val, (int, float))
        and not isinstance(ref_val, bool)
        and not isinstance(cand_val, bool)
    ):
        delta = abs(cand_val - ref_val)
        if delta <= numeric_tolerance:
            return MetadataFieldDifference(
                field_path=field_path,
                metadata_category=category,
                reference_value=ref_val,
                candidate_value=cand_val,
                difference_type=DIFF_UNCHANGED,
                comparison_outcome=OUTCOME_MATCH,
                explanation=f"{field_path} matches within tolerance (ref: {ref_val}{unit_str}, cand: {cand_val}{unit_str}, delta: {round(delta, 4)})",
                is_drift=False,
            )
        else:
            pct_change = ((cand_val - ref_val) / ref_val * 100.0) if ref_val != 0 else 0.0
            sign = "+" if pct_change > 0 else ""
            return MetadataFieldDifference(
                field_path=field_path,
                metadata_category=category,
                reference_value=ref_val,
                candidate_value=cand_val,
                difference_type=DIFF_CHANGED,
                comparison_outcome=OUTCOME_DRIFT,
                explanation=f"{field_path} changed from {ref_val}{unit_str} to {cand_val}{unit_str} (delta: {round(cand_val - ref_val, 4)}, {sign}{round(pct_change, 1)}%)",
                is_drift=True,
            )

    # Exact comparison
    if ref_val == cand_val:
        return MetadataFieldDifference(
            field_path=field_path,
            metadata_category=category,
            reference_value=ref_val,
            candidate_value=cand_val,
            difference_type=DIFF_UNCHANGED,
            comparison_outcome=OUTCOME_MATCH,
            explanation=f"{field_path} matches ({ref_val})",
            is_drift=False,
        )

    # Value differs
    return MetadataFieldDifference(
        field_path=field_path,
        metadata_category=category,
        reference_value=ref_val,
        candidate_value=cand_val,
        difference_type=DIFF_CHANGED,
        comparison_outcome=OUTCOME_DRIFT,
        explanation=f"{field_path} changed from '{ref_val}' to '{cand_val}'",
        is_drift=True,
    )


def compare_tag_dictionaries(
    category_prefix: str,
    category: str,
    ref_tags: Optional[Dict[str, str]],
    cand_tags: Optional[Dict[str, str]],
) -> List[MetadataFieldDifference]:
    """Compares key-value tags between reference and candidate without fabricating defaults."""
    diffs: List[MetadataFieldDifference] = []
    r_tags = ref_tags or {}
    c_tags = cand_tags or {}

    all_keys = sorted(set(r_tags.keys()).union(set(c_tags.keys())))
    if not all_keys:
        diffs.append(
            MetadataFieldDifference(
                field_path=f"{category_prefix}.tags",
                metadata_category=category,
                reference_value=None,
                candidate_value=None,
                difference_type=DIFF_UNCOMPARABLE,
                comparison_outcome=OUTCOME_UNCOMPARABLE,
                explanation=f"No tags present in either reference or candidate for {category_prefix}; tags cannot be compared or positively matched",
                is_drift=False,
            )
        )
        return diffs

    for k in all_keys:
        ref_v = r_tags.get(k)
        cand_v = c_tags.get(k)
        field_p = f"{category_prefix}.tags.{k}"

        if ref_v == cand_v:
            diffs.append(
                MetadataFieldDifference(
                    field_path=field_p,
                    metadata_category=category,
                    reference_value=ref_v,
                    candidate_value=cand_v,
                    difference_type=DIFF_UNCHANGED,
                    comparison_outcome=OUTCOME_MATCH,
                    explanation=f"Tag '{k}' matches ('{ref_v}')",
                    is_drift=False,
                )
            )
        elif ref_v is None:
            diffs.append(
                MetadataFieldDifference(
                    field_path=field_p,
                    metadata_category=category,
                    reference_value=None,
                    candidate_value=cand_v,
                    difference_type=DIFF_ADDED,
                    comparison_outcome=OUTCOME_DRIFT,
                    explanation=f"Tag '{k}' added in candidate (value: '{cand_v}')",
                    is_drift=True,
                )
            )
        elif cand_v is None:
            diffs.append(
                MetadataFieldDifference(
                    field_path=field_p,
                    metadata_category=category,
                    reference_value=ref_v,
                    candidate_value=None,
                    difference_type=DIFF_REMOVED,
                    comparison_outcome=OUTCOME_DRIFT,
                    explanation=f"Tag '{k}' removed in candidate (reference had '{ref_v}')",
                    is_drift=True,
                )
            )
        else:
            diffs.append(
                MetadataFieldDifference(
                    field_path=field_p,
                    metadata_category=category,
                    reference_value=ref_v,
                    candidate_value=cand_v,
                    difference_type=DIFF_CHANGED,
                    comparison_outcome=OUTCOME_DRIFT,
                    explanation=f"Tag '{k}' changed from '{ref_v}' to '{cand_v}'",
                    is_drift=True,
                )
            )
    return diffs


# ==============================================================================
# Deterministic Global Stream Alignment Strategy
# ==============================================================================

def compute_video_stream_affinity(ref: VideoStreamMetadata, cand: VideoStreamMetadata) -> int:
    """Computes a deterministic affinity score between two video streams."""
    score = 0
    # Codec match is a strong fundamental indicator
    if ref.codec_name and cand.codec_name and ref.codec_name.lower() == cand.codec_name.lower():
        score += 40
    # Exact resolution match
    if ref.width is not None and cand.width is not None and ref.width == cand.width:
        score += 20
    if ref.height is not None and cand.height is not None and ref.height == cand.height:
        score += 20
    # Stream index match
    if ref.stream_index == cand.stream_index:
        score += 15
    # Pixel format match
    if ref.pixel_format and cand.pixel_format and ref.pixel_format == cand.pixel_format:
        score += 10
    # Average frame rate match
    if (
        ref.avg_frame_rate is not None
        and cand.avg_frame_rate is not None
        and abs(ref.avg_frame_rate - cand.avg_frame_rate) < 0.05
    ):
        score += 5
    return score


def compute_audio_stream_affinity(ref: AudioStreamMetadata, cand: AudioStreamMetadata) -> int:
    """Computes a deterministic affinity score between two audio streams."""
    score = 0
    # Codec match
    if ref.codec_name and cand.codec_name and ref.codec_name.lower() == cand.codec_name.lower():
        score += 40
    # Sample rate match
    if ref.sample_rate is not None and cand.sample_rate is not None and ref.sample_rate == cand.sample_rate:
        score += 25
    # Channels count match
    if ref.channels is not None and cand.channels is not None and ref.channels == cand.channels:
        score += 20
    # Stream index match
    if ref.stream_index == cand.stream_index:
        score += 15
    # Channel layout match
    if ref.channel_layout and cand.channel_layout and ref.channel_layout == cand.channel_layout:
        score += 10
    return score


def find_all_optimal_matchings(
    ref_count: int,
    cand_count: int,
    affinity_matrix: List[List[int]],
    min_affinity: int = MIN_PLAUSIBLE_AFFINITY,
    max_evaluations: int = DEFAULT_MAX_MATCHING_EVALUATIONS,
    max_optimal_mappings: int = DEFAULT_MAX_OPTIMAL_MAPPINGS,
) -> Tuple[int, int, List[Dict[int, int]], bool]:
    """
    Finds all one-to-one matchings between reference and candidate streams
    that achieve the global maximum total affinity score within a bounded search budget.

    Enforces injective (one-to-one) correspondence: a candidate stream cannot be
    matched to multiple reference streams.

    Complexity bounding:
    - Search state evaluations are capped at max_evaluations.
    - Optimal assignments recorded are capped at max_optimal_mappings.
    - If the computational budget is exceeded, returns bound_exceeded=True to prevent
      arbitrary or incomplete matching decisions.

    Returns:
        (max_score, max_pairs, list_of_optimal_mappings, bound_exceeded)
        where each mapping is {ref_idx: cand_idx}.
    """
    if ref_count == 0 or cand_count == 0:
        return 0, 0, [{}], False

    valid_candidates: List[List[Tuple[int, int]]] = []
    for r in range(ref_count):
        c_list = [
            (c, affinity_matrix[r][c])
            for c in range(cand_count)
            if affinity_matrix[r][c] >= min_affinity
        ]
        # Sort descending by score, deterministic tie-break by candidate index ascending
        c_list.sort(key=lambda x: (x[1], -x[0]), reverse=True)
        valid_candidates.append(c_list)

    # Precalculate upper bounds for branch-and-bound pruning
    max_possible_remaining = [0] * (ref_count + 1)
    for r in range(ref_count - 1, -1, -1):
        max_r = max([score for _, score in valid_candidates[r]], default=0)
        max_possible_remaining[r] = max_possible_remaining[r + 1] + max_r

    best_score = -1
    best_pairs = -1
    best_mappings: List[Dict[int, int]] = []
    evaluations_count = 0
    bound_exceeded = False

    def search(r_idx: int, current_score: int, used_cand: Set[int], current_mapping: Dict[int, int]):
        nonlocal best_score, best_pairs, best_mappings, evaluations_count, bound_exceeded

        if bound_exceeded:
            return

        evaluations_count += 1
        if evaluations_count > max_evaluations:
            bound_exceeded = True
            return

        if len(best_mappings) >= max_optimal_mappings:
            bound_exceeded = True
            return

        # Prune if upper bound cannot reach current best_score
        if current_score + max_possible_remaining[r_idx] < best_score:
            return

        if r_idx == ref_count:
            num_pairs = len(current_mapping)
            if current_score > best_score:
                best_score = current_score
                best_pairs = num_pairs
                best_mappings = [dict(current_mapping)]
            elif current_score == best_score:
                if num_pairs > best_pairs:
                    best_pairs = num_pairs
                    best_mappings = [dict(current_mapping)]
                elif num_pairs == best_pairs:
                    best_mappings.append(dict(current_mapping))
                    if len(best_mappings) >= max_optimal_mappings:
                        bound_exceeded = True
            return

        # Option 1: Match r_idx with available candidate
        for c_idx, score in valid_candidates[r_idx]:
            if c_idx not in used_cand:
                used_cand.add(c_idx)
                current_mapping[r_idx] = c_idx
                search(r_idx + 1, current_score + score, used_cand, current_mapping)
                del current_mapping[r_idx]
                used_cand.remove(c_idx)
                if bound_exceeded:
                    return

        # Option 2: Leave r_idx unmatched
        search(r_idx + 1, current_score, used_cand, current_mapping)

    search(0, 0, set(), {})
    return max(0, best_score), max(0, best_pairs), best_mappings if best_mappings else [{}], bound_exceeded


def match_streams_deterministically(
    stream_type: str,
    ref_streams: List[Union[VideoStreamMetadata, AudioStreamMetadata]],
    cand_streams: List[Union[VideoStreamMetadata, AudioStreamMetadata]],
    max_evaluations: int = DEFAULT_MAX_MATCHING_EVALUATIONS,
    max_optimal_mappings: int = DEFAULT_MAX_OPTIMAL_MAPPINGS,
) -> Tuple[
    List[Tuple[Optional[Any], Optional[Any], str]],
    List[StreamMatchRecord],
]:
    """
    Deterministically aligns reference and candidate streams using a bounded global
    assignment strategy with explicit ambiguity detection.

    Strategy:
    1. Computes pairwise affinity matrix between all reference and candidate streams.
    2. Uses bounded branch-and-bound global search to find all maximum-weight one-to-one matchings.
    3. Analyzes optimal solutions:
       - If search budget exceeded, reports all correspondences as STREAM_AMBIGUOUS.
       - Streams uniquely matched across all optimal solutions are marked STREAM_MATCHED.
       - Streams with competing equally optimal assignments are marked STREAM_AMBIGUOUS.
       - Reference streams with no plausible correspondence are marked STREAM_REMOVED.
       - Candidate streams with no plausible correspondence are marked STREAM_ADDED.

    Returns:
        (stream_tuples, match_records)
        where stream_tuples contains (ref_stream, cand_stream, match_status).
    """
    pairs: List[Tuple[Optional[Any], Optional[Any], str]] = []
    records: List[StreamMatchRecord] = []

    n_ref = len(ref_streams)
    n_cand = len(cand_streams)

    if n_ref == 0 and n_cand == 0:
        return pairs, records

    if n_ref == 0:
        for c in cand_streams:
            pairs.append((None, c, STREAM_ADDED))
            records.append(
                StreamMatchRecord(
                    stream_type=stream_type,
                    reference_stream_index=None,
                    candidate_stream_index=c.stream_index,
                    match_status=STREAM_ADDED,
                    explanation=f"Extra {stream_type} stream added in candidate (cand index {c.stream_index})",
                )
            )
        return pairs, records

    if n_cand == 0:
        for r in ref_streams:
            pairs.append((r, None, STREAM_REMOVED))
            records.append(
                StreamMatchRecord(
                    stream_type=stream_type,
                    reference_stream_index=r.stream_index,
                    candidate_stream_index=None,
                    match_status=STREAM_REMOVED,
                    explanation=f"Reference {stream_type} stream (ref index {r.stream_index}) removed in candidate",
                )
            )
        return pairs, records

    affinity_fn = compute_video_stream_affinity if stream_type == "video" else compute_audio_stream_affinity
    matrix = [[affinity_fn(r, c) for c in cand_streams] for r in ref_streams]

    max_score, max_pairs, optimal_mappings, bound_exceeded = find_all_optimal_matchings(
        n_ref,
        n_cand,
        matrix,
        max_evaluations=max_evaluations,
        max_optimal_mappings=max_optimal_mappings,
    )

    if bound_exceeded:
        # Search space or optimal assignment count exceeded the computational bound.
        # Under the bounded matching contract, never arbitrarily choose an assignment;
        # report correspondence as unresolved (STREAM_AMBIGUOUS).
        for r in ref_streams:
            pairs.append((r, None, STREAM_AMBIGUOUS))
            records.append(
                StreamMatchRecord(
                    stream_type=stream_type,
                    reference_stream_index=r.stream_index,
                    candidate_stream_index=None,
                    match_status=STREAM_AMBIGUOUS,
                    explanation=(
                        f"Stream matching computational work bound ({max_evaluations} evaluations) exceeded; "
                        f"{stream_type} stream correspondence unresolved without arbitrary tie-breaking"
                    ),
                )
            )
        return pairs, records

    # Determine unique vs ambiguous assignments
    # For each reference index r, find all candidate indices it was paired with across optimal mappings
    ref_to_cands: Dict[int, Set[Optional[int]]] = {r: set() for r in range(n_ref)}
    cand_to_refs: Dict[int, Set[Optional[int]]] = {c: set() for c in range(n_cand)}

    for mapping in optimal_mappings:
        for r in range(n_ref):
            c = mapping.get(r)
            ref_to_cands[r].add(c)
            if c is not None:
                cand_to_refs[c].add(r)

    matched_cand_indices: Set[int] = set()

    for r_idx, r in enumerate(ref_streams):
        cand_set = ref_to_cands[r_idx]

        if len(cand_set) == 1:
            chosen_c_idx = next(iter(cand_set))
            if chosen_c_idx is not None and len(cand_to_refs[chosen_c_idx]) == 1:
                # Unambiguous one-to-one match
                c = cand_streams[chosen_c_idx]
                matched_cand_indices.add(chosen_c_idx)
                score = matrix[r_idx][chosen_c_idx]
                pairs.append((r, c, STREAM_MATCHED))
                records.append(
                    StreamMatchRecord(
                        stream_type=stream_type,
                        reference_stream_index=r.stream_index,
                        candidate_stream_index=c.stream_index,
                        match_status=STREAM_MATCHED,
                        explanation=f"{stream_type.capitalize()} stream matched (ref index {r.stream_index} -> cand index {c.stream_index}, affinity score: {score})",
                    )
                )
            elif chosen_c_idx is None:
                # Plausibly removed (no candidate with affinity >= min_affinity)
                pairs.append((r, None, STREAM_REMOVED))
                records.append(
                    StreamMatchRecord(
                        stream_type=stream_type,
                        reference_stream_index=r.stream_index,
                        candidate_stream_index=None,
                        match_status=STREAM_REMOVED,
                        explanation=f"Reference {stream_type} stream (ref index {r.stream_index}) removed in candidate (no plausible match)",
                    )
                )
            else:
                # Ambiguity on candidate side
                pairs.append((r, None, STREAM_AMBIGUOUS))
                records.append(
                    StreamMatchRecord(
                        stream_type=stream_type,
                        reference_stream_index=r.stream_index,
                        candidate_stream_index=None,
                        match_status=STREAM_AMBIGUOUS,
                        explanation=f"Ambiguous {stream_type} stream correspondence for reference index {r.stream_index}; competing global assignments exist",
                    )
                )
        else:
            # Genuinely ambiguous correspondence: multiple equally optimal candidate assignments
            pairs.append((r, None, STREAM_AMBIGUOUS))
            records.append(
                StreamMatchRecord(
                    stream_type=stream_type,
                    reference_stream_index=r.stream_index,
                    candidate_stream_index=None,
                    match_status=STREAM_AMBIGUOUS,
                    explanation=f"Ambiguous {stream_type} stream correspondence for reference index {r.stream_index}; multiple equally optimal assignments exist",
                )
            )

    # Remaining candidate streams not uniquely matched to any reference stream
    for c_idx, c in enumerate(cand_streams):
        if c_idx not in matched_cand_indices:
            # If candidate was involved in ambiguous mappings, do not falsely report as added
            if len(cand_to_refs[c_idx]) > 0:
                continue
            pairs.append((None, c, STREAM_ADDED))
            records.append(
                StreamMatchRecord(
                    stream_type=stream_type,
                    reference_stream_index=None,
                    candidate_stream_index=c.stream_index,
                    match_status=STREAM_ADDED,
                    explanation=f"Extra {stream_type} stream added in candidate (cand index {c.stream_index})",
                )
            )

    return pairs, records


# ==============================================================================
# Input Validation Helper
# ==============================================================================

def validate_comparison_inputs(
    reference_metadata: Any,
    candidate_metadata: Any,
    duration_tolerance_seconds: Any,
    fps_tolerance: Any,
) -> Optional[str]:
    """
    Validates public comparison arguments and nested data structures.
    Returns None if valid, or an explanatory error message.
    """
    if not isinstance(reference_metadata, TechnicalMetadataResult):
        return f"reference_metadata must be an instance of TechnicalMetadataResult, got {type(reference_metadata).__name__}"
    if not isinstance(candidate_metadata, TechnicalMetadataResult):
        return f"candidate_metadata must be an instance of TechnicalMetadataResult, got {type(candidate_metadata).__name__}"

    # Validate duration tolerance
    if (
        duration_tolerance_seconds is None
        or isinstance(duration_tolerance_seconds, bool)
        or not isinstance(duration_tolerance_seconds, (int, float))
    ):
        return f"duration_tolerance_seconds must be a non-negative finite number, got {type(duration_tolerance_seconds).__name__}"
    if math.isnan(duration_tolerance_seconds) or math.isinf(duration_tolerance_seconds):
        return "duration_tolerance_seconds cannot be NaN or Infinite"
    if duration_tolerance_seconds < 0.0:
        return f"duration_tolerance_seconds cannot be negative ({duration_tolerance_seconds})"

    # Validate FPS tolerance
    if (
        fps_tolerance is None
        or isinstance(fps_tolerance, bool)
        or not isinstance(fps_tolerance, (int, float))
    ):
        return f"fps_tolerance must be a non-negative finite number, got {type(fps_tolerance).__name__}"
    if math.isnan(fps_tolerance) or math.isinf(fps_tolerance):
        return "fps_tolerance cannot be NaN or Infinite"
    if fps_tolerance < 0.0:
        return f"fps_tolerance cannot be negative ({fps_tolerance})"

    # Validate nested structures
    for label, res in (("reference_metadata", reference_metadata), ("candidate_metadata", candidate_metadata)):
        if not isinstance(res.video_streams, list):
            return f"{label}.video_streams must be a list, got {type(res.video_streams).__name__}"
        if not isinstance(res.audio_streams, list):
            return f"{label}.audio_streams must be a list, got {type(res.audio_streams).__name__}"
        for s in res.video_streams:
            if not isinstance(s, VideoStreamMetadata):
                return f"All items in {label}.video_streams must be VideoStreamMetadata, got {type(s).__name__}"
        for s in res.audio_streams:
            if not isinstance(s, AudioStreamMetadata):
                return f"All items in {label}.audio_streams must be AudioStreamMetadata, got {type(s).__name__}"
        if res.container is not None and not isinstance(res.container, ContainerMetadata):
            return f"{label}.container must be None or ContainerMetadata, got {type(res.container).__name__}"

    return None


# ==============================================================================
# Public Comparison API
# ==============================================================================

def compare_technical_metadata(
    reference_metadata: TechnicalMetadataResult,
    candidate_metadata: TechnicalMetadataResult,
    duration_tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE_SECONDS,
    fps_tolerance: float = DEFAULT_FPS_TOLERANCE,
) -> MetadataComparisonResult:
    """
    Compares technical metadata between a registered reference and candidate media file.

    Identifies differences resulting from transcoding, re-encoding, resizing,
    stream changes, or metadata stripping without assigning authenticity verdicts.

    Args:
        reference_metadata: Previously registered or reference TechnicalMetadataResult.
        candidate_metadata: Newly submitted or candidate TechnicalMetadataResult.
        duration_tolerance_seconds: Tolerance in seconds for container/stream duration comparison.
        fps_tolerance: Floating tolerance for frame-rate comparisons.

    Returns:
        MetadataComparisonResult containing field-level differences, stream alignments,
        reproducible summary counts, and diagnostic details.
    """
    start_time = time.perf_counter()

    # 1. Strict argument and structure validation
    val_err = validate_comparison_inputs(
        reference_metadata,
        candidate_metadata,
        duration_tolerance_seconds,
        fps_tolerance,
    )
    if val_err:
        return MetadataComparisonResult(
            available=False,
            status=STATUS_INVALID_INPUT,
            processing_time_seconds=time.perf_counter() - start_time,
            details={"error": val_err},
        )

    # 2. Extraction availability check
    if not reference_metadata.available or not candidate_metadata.available:
        missing_reasons = []
        if not reference_metadata.available:
            missing_reasons.append(f"Reference metadata unavailable (status={reference_metadata.status})")
        if not candidate_metadata.available:
            missing_reasons.append(f"Candidate metadata unavailable (status={candidate_metadata.status})")

        return MetadataComparisonResult(
            available=False,
            status=STATUS_INSUFFICIENT_EVIDENCE,
            reference_file_path=reference_metadata.file_path,
            candidate_file_path=candidate_metadata.file_path,
            processing_time_seconds=time.perf_counter() - start_time,
            details={"error": "; ".join(missing_reasons)},
        )

    # 3. Check for required comparison evidence (Container presence)
    ref_container = reference_metadata.container
    cand_container = candidate_metadata.container
    if ref_container is None or cand_container is None:
        missing_required = []
        if ref_container is None:
            missing_required.append("Reference container metadata is missing")
        if cand_container is None:
            missing_required.append("Candidate container metadata is missing")
        return MetadataComparisonResult(
            available=False,
            status=STATUS_INSUFFICIENT_EVIDENCE,
            reference_file_path=reference_metadata.file_path,
            candidate_file_path=candidate_metadata.file_path,
            processing_time_seconds=time.perf_counter() - start_time,
            details={"error": "; ".join(missing_required)},
        )

    differences: List[MetadataFieldDifference] = []
    has_critical_drift = False
    has_transcoding_drift = False

    # 4. Compare Container Properties
    # A. Format name
    ref_fmt = ref_container.format_name
    cand_fmt = cand_container.format_name
    fmt_diff_type, fmt_outcome, fmt_exp, fmt_drift = compare_format_names(ref_fmt, cand_fmt)
    differences.append(
        MetadataFieldDifference(
            field_path="container.format_name",
            metadata_category="container",
            reference_value=ref_fmt,
            candidate_value=cand_fmt,
            difference_type=fmt_diff_type,
            comparison_outcome=fmt_outcome,
            explanation=fmt_exp,
            is_drift=fmt_drift,
        )
    )
    if fmt_drift:
        has_transcoding_drift = True

    # B. Container duration
    ref_dur = ref_container.duration_seconds
    cand_dur = cand_container.duration_seconds
    dur_diff = compare_scalar_field(
        "container.duration_seconds",
        "container",
        ref_dur,
        cand_dur,
        numeric_tolerance=duration_tolerance_seconds,
        field_unit="s",
    )
    differences.append(dur_diff)
    if dur_diff.is_drift:
        if ref_dur is not None and cand_dur is not None and abs(cand_dur - ref_dur) > 1.0:
            has_critical_drift = True
        else:
            has_transcoding_drift = True

    # C. Container overall bitrate
    ref_br = ref_container.bitrate
    cand_br = cand_container.bitrate
    br_diff = compare_scalar_field(
        "container.bitrate",
        "container",
        ref_br,
        cand_br,
        field_unit="bps",
    )
    differences.append(br_diff)
    if br_diff.is_drift:
        has_transcoding_drift = True

    # D. Container start time
    ref_st = ref_container.start_time_seconds
    cand_st = cand_container.start_time_seconds
    differences.append(
        compare_scalar_field(
            "container.start_time_seconds",
            "container",
            ref_st,
            cand_st,
            numeric_tolerance=duration_tolerance_seconds,
            field_unit="s",
        )
    )

    # E. Container size
    ref_sz = ref_container.size_bytes
    cand_sz = cand_container.size_bytes
    differences.append(
        compare_scalar_field(
            "container.size_bytes",
            "container",
            ref_sz,
            cand_sz,
            field_unit="bytes",
        )
    )

    # F. Container encoder
    ref_enc = ref_container.encoder
    cand_enc = cand_container.encoder
    enc_diff = compare_scalar_field(
        "container.encoder",
        "container",
        ref_enc,
        cand_enc,
    )
    differences.append(enc_diff)
    if enc_diff.is_drift:
        has_transcoding_drift = True

    # G. Container tags
    ref_c_tags = ref_container.tags
    cand_c_tags = cand_container.tags
    differences.extend(compare_tag_dictionaries("container", "container", ref_c_tags, cand_c_tags))

    # 5. Stream Counts Comparison
    differences.append(
        compare_scalar_field(
            "stream_count.video",
            "stream_count",
            reference_metadata.video_stream_count,
            candidate_metadata.video_stream_count,
        )
    )
    if reference_metadata.video_stream_count != candidate_metadata.video_stream_count:
        has_critical_drift = True

    differences.append(
        compare_scalar_field(
            "stream_count.audio",
            "stream_count",
            reference_metadata.audio_stream_count,
            candidate_metadata.audio_stream_count,
        )
    )
    if reference_metadata.audio_stream_count != candidate_metadata.audio_stream_count:
        has_critical_drift = True

    # 6. Video Streams Comparison
    all_stream_matches: List[StreamMatchRecord] = []
    has_unresolved_correspondence = False

    video_pairs, video_match_records = match_streams_deterministically(
        "video",
        reference_metadata.video_streams,
        candidate_metadata.video_streams,
    )
    all_stream_matches.extend(video_match_records)

    added_streams_cnt = 0
    removed_streams_cnt = 0

    for r_v, c_v, match_status in video_pairs:
        if match_status == STREAM_MATCHED and r_v is not None and c_v is not None:
            idx_label = f"video[ref={r_v.stream_index},cand={c_v.stream_index}]"

            # Codec name
            c_codec = compare_scalar_field(f"{idx_label}.codec_name", "video", r_v.codec_name, c_v.codec_name)
            differences.append(c_codec)
            if c_codec.is_drift:
                has_transcoding_drift = True

            # Profile & level
            differences.append(compare_scalar_field(f"{idx_label}.profile", "video", r_v.profile, c_v.profile))
            differences.append(compare_scalar_field(f"{idx_label}.level", "video", r_v.level, c_v.level))

            # Pixel format
            c_pix = compare_scalar_field(f"{idx_label}.pixel_format", "video", r_v.pixel_format, c_v.pixel_format)
            differences.append(c_pix)
            if c_pix.is_drift:
                has_transcoding_drift = True

            # Dimensions (Width & Height) -> Critical drift if modified
            c_w = compare_scalar_field(f"{idx_label}.width", "video", r_v.width, c_v.width, field_unit="px")
            c_h = compare_scalar_field(f"{idx_label}.height", "video", r_v.height, c_v.height, field_unit="px")
            differences.append(c_w)
            differences.append(c_h)
            if c_w.is_drift or c_h.is_drift:
                has_critical_drift = True

            # Aspect ratios
            differences.append(compare_scalar_field(f"{idx_label}.sample_aspect_ratio", "video", r_v.sample_aspect_ratio, c_v.sample_aspect_ratio))
            differences.append(compare_scalar_field(f"{idx_label}.display_aspect_ratio", "video", r_v.display_aspect_ratio, c_v.display_aspect_ratio))

            # Frame rates
            differences.append(
                compare_scalar_field(
                    f"{idx_label}.avg_frame_rate",
                    "video",
                    r_v.avg_frame_rate,
                    c_v.avg_frame_rate,
                    numeric_tolerance=fps_tolerance,
                    field_unit="fps",
                )
            )
            differences.append(
                compare_scalar_field(
                    f"{idx_label}.r_frame_rate",
                    "video",
                    r_v.r_frame_rate,
                    c_v.r_frame_rate,
                    numeric_tolerance=fps_tolerance,
                    field_unit="fps",
                )
            )

            # Color properties
            differences.append(compare_scalar_field(f"{idx_label}.color_range", "video", r_v.color_range, c_v.color_range))
            differences.append(compare_scalar_field(f"{idx_label}.color_space", "video", r_v.color_space, c_v.color_space))
            differences.append(compare_scalar_field(f"{idx_label}.color_transfer", "video", r_v.color_transfer, c_v.color_transfer))
            differences.append(compare_scalar_field(f"{idx_label}.color_primaries", "video", r_v.color_primaries, c_v.color_primaries))

            # Frame count
            differences.append(compare_scalar_field(f"{idx_label}.frame_count", "video", r_v.frame_count, c_v.frame_count, field_unit="frames"))

            # Time base
            differences.append(compare_scalar_field(f"{idx_label}.time_base", "video", r_v.time_base, c_v.time_base))

            # Stream duration & bitrate
            differences.append(
                compare_scalar_field(
                    f"{idx_label}.duration_seconds",
                    "video",
                    r_v.duration_seconds,
                    c_v.duration_seconds,
                    numeric_tolerance=duration_tolerance_seconds,
                    field_unit="s",
                )
            )
            differences.append(compare_scalar_field(f"{idx_label}.bitrate", "video", r_v.bitrate, c_v.bitrate, field_unit="bps"))

            # Stream tags
            differences.extend(compare_tag_dictionaries(idx_label, "video", r_v.raw_tags, c_v.raw_tags))

        elif match_status == STREAM_REMOVED and r_v is not None:
            removed_streams_cnt += 1
            has_critical_drift = True
            differences.append(
                MetadataFieldDifference(
                    field_path=f"video[ref={r_v.stream_index}]",
                    metadata_category="video",
                    reference_value=r_v.to_dict(),
                    candidate_value=None,
                    difference_type=DIFF_REMOVED,
                    comparison_outcome=OUTCOME_DRIFT,
                    explanation=f"Reference video stream (index {r_v.stream_index}, {r_v.codec_name}) has no counterpart in candidate",
                    is_drift=True,
                )
            )
        elif match_status == STREAM_ADDED and c_v is not None:
            added_streams_cnt += 1
            has_critical_drift = True
            differences.append(
                MetadataFieldDifference(
                    field_path=f"video[cand={c_v.stream_index}]",
                    metadata_category="video",
                    reference_value=None,
                    candidate_value=c_v.to_dict(),
                    difference_type=DIFF_ADDED,
                    comparison_outcome=OUTCOME_DRIFT,
                    explanation=f"Extra candidate video stream (index {c_v.stream_index}, {c_v.codec_name}) added",
                    is_drift=True,
                )
            )
        elif match_status == STREAM_AMBIGUOUS and r_v is not None:
            has_unresolved_correspondence = True
            differences.append(
                MetadataFieldDifference(
                    field_path=f"video[ref={r_v.stream_index}]",
                    metadata_category="video",
                    reference_value=r_v.to_dict(),
                    candidate_value=None,
                    difference_type=DIFF_UNCOMPARABLE,
                    comparison_outcome=OUTCOME_UNCOMPARABLE,
                    explanation=f"Reference video stream (index {r_v.stream_index}) has ambiguous candidate correspondences",
                    is_drift=False,
                )
            )

    # 7. Audio Streams Comparison
    audio_pairs, audio_match_records = match_streams_deterministically(
        "audio",
        reference_metadata.audio_streams,
        candidate_metadata.audio_streams,
    )
    all_stream_matches.extend(audio_match_records)

    for r_a, c_a, match_status in audio_pairs:
        if match_status == STREAM_MATCHED and r_a is not None and c_a is not None:
            idx_label = f"audio[ref={r_a.stream_index},cand={c_a.stream_index}]"

            # Codec name
            c_acodec = compare_scalar_field(f"{idx_label}.codec_name", "audio", r_a.codec_name, c_a.codec_name)
            differences.append(c_acodec)
            if c_acodec.is_drift:
                has_transcoding_drift = True

            # Sample rate
            c_sr = compare_scalar_field(f"{idx_label}.sample_rate", "audio", r_a.sample_rate, c_a.sample_rate, field_unit="Hz")
            differences.append(c_sr)
            if c_sr.is_drift:
                has_transcoding_drift = True

            # Channel count & layout
            differences.append(compare_scalar_field(f"{idx_label}.channels", "audio", r_a.channels, c_a.channels))
            differences.append(compare_scalar_field(f"{idx_label}.channel_layout", "audio", r_a.channel_layout, c_a.channel_layout))

            # Duration & bitrate
            differences.append(
                compare_scalar_field(
                    f"{idx_label}.duration_seconds",
                    "audio",
                    r_a.duration_seconds,
                    c_a.duration_seconds,
                    numeric_tolerance=duration_tolerance_seconds,
                    field_unit="s",
                )
            )
            differences.append(compare_scalar_field(f"{idx_label}.bitrate", "audio", r_a.bitrate, c_a.bitrate, field_unit="bps"))

            # Audio tags
            differences.extend(compare_tag_dictionaries(idx_label, "audio", r_a.raw_tags, c_a.raw_tags))

        elif match_status == STREAM_REMOVED and r_a is not None:
            removed_streams_cnt += 1
            has_critical_drift = True
            differences.append(
                MetadataFieldDifference(
                    field_path=f"audio[ref={r_a.stream_index}]",
                    metadata_category="audio",
                    reference_value=r_a.to_dict(),
                    candidate_value=None,
                    difference_type=DIFF_REMOVED,
                    comparison_outcome=OUTCOME_DRIFT,
                    explanation=f"Reference audio stream (index {r_a.stream_index}, {r_a.codec_name}) removed in candidate",
                    is_drift=True,
                )
            )
        elif match_status == STREAM_ADDED and c_a is not None:
            added_streams_cnt += 1
            has_critical_drift = True
            differences.append(
                MetadataFieldDifference(
                    field_path=f"audio[cand={c_a.stream_index}]",
                    metadata_category="audio",
                    reference_value=None,
                    candidate_value=c_a.to_dict(),
                    difference_type=DIFF_ADDED,
                    comparison_outcome=OUTCOME_DRIFT,
                    explanation=f"Extra candidate audio stream (index {c_a.stream_index}, {c_a.codec_name}) added",
                    is_drift=True,
                )
            )
        elif match_status == STREAM_AMBIGUOUS and r_a is not None:
            has_unresolved_correspondence = True
            differences.append(
                MetadataFieldDifference(
                    field_path=f"audio[ref={r_a.stream_index}]",
                    metadata_category="audio",
                    reference_value=r_a.to_dict(),
                    candidate_value=None,
                    difference_type=DIFF_UNCOMPARABLE,
                    comparison_outcome=OUTCOME_UNCOMPARABLE,
                    explanation=f"Reference audio stream (index {r_a.stream_index}) has ambiguous candidate correspondences",
                    is_drift=False,
                )
            )

    # 8. Compute Reproducible Summary Counts (Strict Partition Invariant)
    unchanged_cnt = sum(1 for d in differences if d.difference_type == DIFF_UNCHANGED)
    changed_cnt = sum(1 for d in differences if d.difference_type in (DIFF_CHANGED, DIFF_ADDED, DIFF_REMOVED))
    missing_cnt = sum(1 for d in differences if d.difference_type in (DIFF_MISSING_REFERENCE_VALUE, DIFF_MISSING_CANDIDATE_VALUE))
    uncomparable_cnt = sum(1 for d in differences if d.difference_type == DIFF_UNCOMPARABLE)
    jointly_unavailable_cnt = sum(
        1 for d in differences
        if d.difference_type == DIFF_UNCOMPARABLE and d.reference_value is None and d.candidate_value is None
    )
    total_compared = len(differences)

    # Invariant assertion: every field difference belongs to exactly one partition
    assert total_compared == unchanged_cnt + changed_cnt + missing_cnt + uncomparable_cnt

    summary = MetadataComparisonSummary(
        total_compared_fields=total_compared,
        unchanged_fields_count=unchanged_cnt,
        changed_fields_count=changed_cnt,
        missing_fields_count=missing_cnt,
        uncomparable_fields_count=uncomparable_cnt,
        jointly_unavailable_fields_count=jointly_unavailable_cnt,
        added_streams_count=added_streams_cnt,
        removed_streams_count=removed_streams_cnt,
        has_critical_drift=has_critical_drift,
        has_transcoding_drift=has_transcoding_drift,
    )

    # 9. Determine Overall Status Under Documented Policy
    # Policy:
    # - INSUFFICIENT_EVIDENCE: Unresolved correspondences or missing required comparison evidence
    # - SUCCESS_WITH_DRIFT: Actual differences observed (not triggered solely by missing optional fields)
    # - SUCCESS_WITH_MISSING_FIELDS: No differences observed, but optional metadata values are missing/jointly unavailable
    # - SUCCESS: Complete comparison completed without observed drift, missing values, or uncomparable properties
    if has_unresolved_correspondence:
        overall_status = STATUS_INSUFFICIENT_EVIDENCE
    elif changed_cnt > 0:
        overall_status = STATUS_SUCCESS_WITH_DRIFT
    elif missing_cnt > 0 or uncomparable_cnt > 0:
        overall_status = STATUS_SUCCESS_WITH_MISSING_FIELDS
    else:
        overall_status = STATUS_SUCCESS

    elapsed = time.perf_counter() - start_time

    return MetadataComparisonResult(
        available=True,
        status=overall_status,
        differences=differences,
        summary=summary,
        stream_matches=all_stream_matches,
        reference_file_path=reference_metadata.file_path,
        candidate_file_path=candidate_metadata.file_path,
        processing_time_seconds=elapsed,
        details={
            "video_stream_count_delta": candidate_metadata.video_stream_count - reference_metadata.video_stream_count,
            "audio_stream_count_delta": candidate_metadata.audio_stream_count - reference_metadata.audio_stream_count,
            "stream_match_count": len(all_stream_matches),
            "has_unresolved_correspondence": has_unresolved_correspondence,
        },
    )
