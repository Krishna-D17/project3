"""
Deterministic Temporal Perceptual Difference-Hash (dHash) Extraction.

Implements Layer 1 (Temporal Perceptual Fingerprinting) for MediaProof.
Extracts an ordered sequence of 64-bit perceptual difference hashes across
deterministic video timestamps for tamper and edit-alignment analysis.

Algorithm Specification:
1. Frame Sampling:
   - Evaluates video media at a deterministic, configurable interval (default: 1.0s).
   - Samples start at t = 0.0s (first frame) and advance at regular intervals.
2. Frame Preprocessing:
   - Converts frame to 8-bit grayscale.
   - Resizes grayscale frame to exactly 9 columns (width) by 8 rows (height)
     using deterministic pixel area relation interpolation (cv2.INTER_AREA).
3. Horizontal Difference Hashing:
   - For each row r in 0..7 and column c in 0..7:
       left_pixel  = image[r, c]
       right_pixel = image[r, c + 1]
       bit = 1 if left_pixel > right_pixel else 0
   - Exactly 8 comparisons per row across 8 rows = 64 comparisons (64 bits).
4. Bit Ordering & Formatting:
   - Row-major traversal (top-to-bottom, left-to-right).
   - MSB-first: (row 0, col 0) is bit 63; (row 7, col 7) is bit 0.
   - Hexadecimal representation: 16-character lowercase hex string with leading zeros.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Try importing cv2 safely
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

ALGORITHM_NAME = "temporal_dhash"
ALGORITHM_VERSION = "1.0.0"
DEFAULT_SAMPLING_INTERVAL_SECONDS = 1.0
DHASH_WIDTH = 9
DHASH_HEIGHT = 8
DHASH_BITS = 64
DHASH_INTERPOLATION = "cv2.INTER_AREA"
DHASH_BIT_ORDERING = "row_major_left_to_right_msb_first"

# Status Constants
STATUS_SUCCESS = "SUCCESS"
STATUS_SUCCESS_WITH_SKIPPED = "SUCCESS_WITH_SKIPPED_FRAMES"
STATUS_UNAVAILABLE_DECODER = "UNAVAILABLE_DECODER_MISSING"
STATUS_INVALID_MEDIA = "INVALID_MEDIA"
STATUS_ZERO_SAMPLES = "ZERO_USABLE_SAMPLES"
STATUS_DECODE_FAILURE = "DECODE_FAILURE"
STATUS_INVALID_INTERVAL = "INVALID_SAMPLING_INTERVAL"


@dataclass
class TemporalDHashSample:
    """A single difference-hash sample from a video frame at a specific timestamp."""
    sample_index: int
    timestamp_seconds: float
    dhash_hex: str
    dhash_int: int
    frame_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "timestamp_seconds": round(self.timestamp_seconds, 4),
            "frame_index": self.frame_index,
            "dhash_hex": self.dhash_hex,
            "dhash_int": self.dhash_int,
        }


@dataclass
class TemporalDHashResult:
    """Structured result of temporal perceptual difference-hash extraction."""
    available: bool
    status: str
    algorithm: str = f"{ALGORITHM_NAME}_v{ALGORITHM_VERSION}"
    sampling_interval_seconds: float = DEFAULT_SAMPLING_INTERVAL_SECONDS
    samples: List[TemporalDHashSample] = field(default_factory=list)
    sample_count: int = 0
    skipped_samples_count: int = 0
    processing_time_seconds: float = 0.0
    video_duration_seconds: Optional[float] = None
    video_fps: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "status": self.status,
            "algorithm": self.algorithm,
            "sampling_interval_seconds": self.sampling_interval_seconds,
            "sample_count": self.sample_count,
            "skipped_samples_count": self.skipped_samples_count,
            "samples": [s.to_dict() for s in self.samples],
            "processing_time_seconds": round(self.processing_time_seconds, 4),
            "video_duration_seconds": (
                round(self.video_duration_seconds, 4)
                if self.video_duration_seconds is not None
                else None
            ),
            "video_fps": (
                round(self.video_fps, 4) if self.video_fps is not None else None
            ),
            "details": self.details,
        }


def compute_frame_dhash(
    frame: Union[np.ndarray, Any],
) -> Tuple[int, str]:
    """
    Computes a deterministic 64-bit difference hash (dHash) from an image or video frame.

    Args:
        frame: 2D or 3D numpy array or PIL Image representing the image frame.

    Returns:
        Tuple of (64-bit integer, 16-character lowercase hex string).
    """
    # 1. Normalize input to grayscale numpy array
    if hasattr(frame, "convert"):  # PIL Image
        gray = np.array(frame.convert("L"), dtype=np.uint8)
    elif isinstance(frame, np.ndarray):
        if frame.ndim == 3:
            # Assume BGR if OpenCV convention, check channel count
            if frame.shape[2] == 3:
                if HAS_CV2:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    # Standard ITU-R 601-2 luma fallback: Y = 0.299 R + 0.587 G + 0.114 B
                    gray = np.dot(frame[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
            elif frame.shape[2] == 4:
                if HAS_CV2:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                else:
                    gray = np.dot(frame[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
            elif frame.shape[2] == 1:
                gray = frame.squeeze(axis=2)
            else:
                raise ValueError(f"Unsupported number of image channels: {frame.shape[2]}")
        elif frame.ndim == 2:
            gray = frame.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported image array dimensions: {frame.ndim}")
    else:
        raise TypeError(f"Unsupported image frame type: {type(frame)}")

    # 2. Resize to 9 (width) x 8 (height)
    if gray.shape != (DHASH_HEIGHT, DHASH_WIDTH):
        if HAS_CV2:
            resized = cv2.resize(
                gray, (DHASH_WIDTH, DHASH_HEIGHT), interpolation=cv2.INTER_AREA
            )
        else:
            from PIL import Image
            pil_img = Image.fromarray(gray)
            resized = np.array(
                pil_img.resize((DHASH_WIDTH, DHASH_HEIGHT), Image.Resampling.BILINEAR),
                dtype=np.uint8
            )
    else:
        resized = gray

    # 3. Compute horizontal difference: left > right
    # Shape: (8, 8)
    diff = resized[:, :8] > resized[:, 1:]

    # 4. Pack 64 boolean comparisons into unsigned 64-bit integer (MSB first)
    hash_int = 0
    for bit in diff.flatten():
        hash_int = (hash_int << 1) | int(bit)

    # 5. Format as 16-character lowercase hex string
    hash_hex = f"{hash_int:016x}"
    return hash_int, hash_hex


def extract_temporal_dhash(
    file_path: Union[str, Path],
    sampling_interval_seconds: float = DEFAULT_SAMPLING_INTERVAL_SECONDS,
    max_samples: Optional[int] = None,
) -> TemporalDHashResult:
    """
    Extracts deterministic temporal difference hashes from a video at regular intervals.

    Args:
        file_path: Path to the video file.
        sampling_interval_seconds: Interval in seconds between sampled frames (must be > 0).
        max_samples: Optional upper bound on the number of samples extracted.

    Returns:
        Structured TemporalDHashResult.
    """
    start_time = time.time()

    # 1. Validate sampling interval
    if not isinstance(sampling_interval_seconds, (int, float)) or math.isnan(sampling_interval_seconds) or math.isinf(sampling_interval_seconds) or sampling_interval_seconds <= 0:
        return TemporalDHashResult(
            available=False,
            status=STATUS_INVALID_INTERVAL,
            sampling_interval_seconds=float(sampling_interval_seconds) if isinstance(sampling_interval_seconds, (int, float)) else -1.0,
            processing_time_seconds=time.time() - start_time,
            details={"error": f"sampling_interval_seconds must be a positive finite number, got {sampling_interval_seconds}."}
        )

    # 2. Validate media file existence
    p = Path(file_path).resolve()
    if not p.exists() or not p.is_file():
        return TemporalDHashResult(
            available=False,
            status=STATUS_INVALID_MEDIA,
            sampling_interval_seconds=sampling_interval_seconds,
            processing_time_seconds=time.time() - start_time,
            details={"error": f"Video file not found or is not a regular file: '{p}'."}
        )

    # 3. Check decoder backend availability
    if not HAS_CV2 or cv2 is None:
        return TemporalDHashResult(
            available=False,
            status=STATUS_UNAVAILABLE_DECODER,
            sampling_interval_seconds=sampling_interval_seconds,
            processing_time_seconds=time.time() - start_time,
            details={"error": "OpenCV (cv2) video decoding backend is not installed or available."}
        )

    # 4. Open video capture
    try:
        cap = cv2.VideoCapture(str(p))
    except Exception as e:
        return TemporalDHashResult(
            available=False,
            status=STATUS_INVALID_MEDIA,
            sampling_interval_seconds=sampling_interval_seconds,
            processing_time_seconds=time.time() - start_time,
            details={"error": f"Failed to initialize video capture: {str(e)}"}
        )

    if not cap.isOpened():
        cap.release()
        return TemporalDHashResult(
            available=False,
            status=STATUS_INVALID_MEDIA,
            sampling_interval_seconds=sampling_interval_seconds,
            processing_time_seconds=time.time() - start_time,
            details={"error": "Decoder could not open or parse media stream."}
        )

    try:
        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_raw) if (fps_raw and not math.isnan(fps_raw) and fps_raw > 0) else None

        frame_count_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = int(frame_count_raw) if (frame_count_raw and not math.isnan(frame_count_raw) and frame_count_raw > 0) else None

        duration_sec = (total_frames / fps) if (fps is not None and total_frames is not None and fps > 0) else None

        samples: List[TemporalDHashSample] = []
        skipped_count = 0
        sample_idx = 0

        # Strategy A: Seek-based sampling if FPS and total_frames are available
        if fps is not None and fps > 0 and total_frames is not None and total_frames > 0:
            target_time = 0.0
            last_sampled_frame = -1

            while target_time < duration_sec or (target_time == 0.0 and total_frames > 0):
                if max_samples is not None and sample_idx >= max_samples:
                    break

                target_frame_idx = min(round(target_time * fps), total_frames - 1)

                # Avoid duplicate extraction of the same frame index for short clips
                if target_frame_idx == last_sampled_frame and sample_idx > 0:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                ret, frame = cap.read()

                if ret and frame is not None:
                    actual_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    actual_frame_idx = max(0, actual_frame_idx)
                    d_int, d_hex = compute_frame_dhash(frame)
                    samples.append(TemporalDHashSample(
                        sample_index=sample_idx,
                        timestamp_seconds=target_time,
                        dhash_hex=d_hex,
                        dhash_int=d_int,
                        frame_index=actual_frame_idx,
                    ))
                    sample_idx += 1
                    last_sampled_frame = actual_frame_idx
                else:
                    skipped_count += 1

                target_time += sampling_interval_seconds

        # Strategy B: Sequential read if seeking/container metadata is incomplete
        else:
            current_frame_idx = 0
            next_target_sec = 0.0

            while True:
                if max_samples is not None and sample_idx >= max_samples:
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Calculate current timestamp from msec property or frame index fallback
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_msec and not math.isnan(pos_msec) and pos_msec >= 0:
                    current_sec = pos_msec / 1000.0
                elif fps is not None and fps > 0:
                    current_sec = current_frame_idx / fps
                else:
                    current_sec = float(current_frame_idx)

                if current_sec >= next_target_sec or current_frame_idx == 0:
                    d_int, d_hex = compute_frame_dhash(frame)
                    samples.append(TemporalDHashSample(
                        sample_index=sample_idx,
                        timestamp_seconds=round(current_sec, 4),
                        dhash_hex=d_hex,
                        dhash_int=d_int,
                        frame_index=current_frame_idx,
                    ))
                    sample_idx += 1
                    next_target_sec = current_sec + sampling_interval_seconds

                current_frame_idx += 1

    except Exception as e:
        cap.release()
        return TemporalDHashResult(
            available=False,
            status=STATUS_DECODE_FAILURE,
            sampling_interval_seconds=sampling_interval_seconds,
            processing_time_seconds=time.time() - start_time,
            details={"error": f"Exception encountered during frame decoding: {str(e)}"}
        )
    finally:
        cap.release()

    total_proc_time = time.time() - start_time

    # Determine extraction status
    if len(samples) == 0:
        return TemporalDHashResult(
            available=False,
            status=STATUS_ZERO_SAMPLES,
            sampling_interval_seconds=sampling_interval_seconds,
            samples=[],
            sample_count=0,
            skipped_samples_count=skipped_count,
            processing_time_seconds=total_proc_time,
            video_duration_seconds=duration_sec,
            video_fps=fps,
            details={
                "error": "Video decoding returned zero usable frames.",
                "total_frames_reported": total_frames,
            }
        )

    final_status = (
        STATUS_SUCCESS_WITH_SKIPPED if skipped_count > 0 else STATUS_SUCCESS
    )

    return TemporalDHashResult(
        available=True,
        status=final_status,
        sampling_interval_seconds=sampling_interval_seconds,
        samples=samples,
        sample_count=len(samples),
        skipped_samples_count=skipped_count,
        processing_time_seconds=total_proc_time,
        video_duration_seconds=duration_sec,
        video_fps=fps,
        details={
            "decoder": "opencv",
            "dimensions": [DHASH_WIDTH, DHASH_HEIGHT],
            "interpolation": DHASH_INTERPOLATION,
            "bit_ordering": DHASH_BIT_ORDERING,
            "total_frames": total_frames,
        }
    )
