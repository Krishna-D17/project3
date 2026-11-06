import shutil
import tempfile
from pathlib import Path
from typing import List, Union

import numpy as np

from media.ffmpeg import get_ffmpeg_path, run_ffmpeg_command
from media.service import MediaService
from forensics.config import (
    CATEGORY_GENERIC_TECHNICAL,
    HEURISTIC_BLURRINESS_LAPLACIAN_THRESHOLD,
    HEURISTIC_FFT_HIGH_FREQ_RATIO_THRESHOLD,
    HEURISTIC_TEMPORAL_NOISE_VAR_THRESHOLD,
    MAX_VIDEO_SAMPLED_FRAMES,
    RISK_LEVEL_HIGH,
    RISK_LEVEL_LOW,
    RISK_LEVEL_MEDIUM,
    VIDEO_TARGET_FPS_SAMPLING,
)
from forensics.models import ForensicSignal


def extract_video_signals(file_path: Union[str, Path], timeout: float = 30.0) -> List[ForensicSignal]:
    """Extracts deterministic spatial and temporal forensic features from video frames."""
    signals: List[ForensicSignal] = []
    p = Path(file_path).resolve()

    ffmpeg_exe = get_ffmpeg_path()
    if not ffmpeg_exe or not ffmpeg_exe.exists():
        signals.append(ForensicSignal(
            signal_name="video_forensics_status",
            risk_level=RISK_LEVEL_MEDIUM,
            score=0.5,
            details="FFmpeg binary not available for video frame extraction.",
            evidence_strength="low",
            type="heuristic",
            category=CATEGORY_GENERIC_TECHNICAL,
        ))
        return signals

    probe_res = MediaService.probe_file(p)
    if not probe_res.has_video:
        return signals

    temp_dir = Path(tempfile.mkdtemp(prefix="mp_forensics_video_"))

    try:
        # 1. Extract sampled frames as 8-bit PGM/PPM or grayscale raw frames via FFmpeg
        # We sample 1 fps up to MAX_VIDEO_SAMPLED_FRAMES
        frame_pattern = temp_dir / "frame_%03d.pgm"
        cmd_extract = [
            str(ffmpeg_exe), "-y",
            "-i", str(p),
            "-vf", f"fps={VIDEO_TARGET_FPS_SAMPLING},scale=256:256",
            "-vframes", str(MAX_VIDEO_SAMPLED_FRAMES),
            str(frame_pattern)
        ]
        res_ext = run_ffmpeg_command(cmd_extract, timeout=timeout)
        if res_ext.returncode != 0:
            signals.append(ForensicSignal(
                signal_name="frame_extraction_status",
                risk_level=RISK_LEVEL_MEDIUM,
                score=0.5,
                details=f"FFmpeg frame sampling failed: {res_ext.stderr}",
                evidence_strength="low",
                type="heuristic"
            ))
            return signals

        frame_files = sorted(temp_dir.glob("frame_*.pgm"))
        if not frame_files:
            signals.append(ForensicSignal(
                signal_name="sampled_frames_count",
                risk_level=RISK_LEVEL_LOW,
                score=0.0,
                details="No frames extracted from video stream.",
                evidence_strength="low",
                type="heuristic"
            ))
            return signals

        # 2. Parse Netpbm PGM frames into 2D NumPy arrays (256x256 grayscale)
        frames_data = []
        for f_path in frame_files:
            try:
                with open(f_path, "rb") as f:
                    header = f.readline().decode("ascii", errors="ignore").strip()
                    if header not in ["P5", "P2"]:
                        continue
                    line = f.readline().decode("ascii", errors="ignore").strip()
                    while line.startswith("#"):
                        line = f.readline().decode("ascii", errors="ignore").strip()
                    dims = line.split()
                    w, h = int(dims[0]), int(dims[1])
                    maxval = int(f.readline().decode("ascii", errors="ignore").strip())
                    raw_buf = f.read(w * h)
                    arr = np.frombuffer(raw_buf, dtype=np.uint8).reshape((h, w)).astype(np.float32)
                    frames_data.append(arr)
            except Exception:
                pass

        if not frames_data:
            signals.append(ForensicSignal(
                signal_name="frame_parser_status",
                risk_level=RISK_LEVEL_LOW,
                score=0.0,
                details="Could not parse sampled PGM frame buffers.",
                evidence_strength="low",
                type="heuristic"
            ))
            return signals

        n_frames = len(frames_data)

        # Signal A: Frame Sample Count
        signals.append(ForensicSignal(
            signal_name="video_sampled_frame_count",
            risk_level=RISK_LEVEL_LOW,
            score=0.0,
            details=f"Analyzed {n_frames} sampled frames at {VIDEO_TARGET_FPS_SAMPLING} fps.",
            evidence_strength="low",
            type="heuristic"
        ))

        # Signal B: 2D FFT High-Frequency Energy Ratio (Spatial Grid Artifacts)
        fft_high_ratios = []
        blurriness_scores = []

        for frame in frames_data:
            # 2D FFT
            f_transform = np.fft.fft2(frame)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)

            rows, cols = frame.shape
            crow, ccol = rows // 2, cols // 2

            # Define low frequency center mask (radius 32 px)
            r = 32
            mask = np.zeros((rows, cols), dtype=bool)
            mask[crow - r:crow + r, ccol - r:ccol + r] = True

            total_energy = float(np.sum(magnitude)) + 1e-8
            high_freq_energy = float(np.sum(magnitude[~mask]))
            ratio = high_freq_energy / total_energy
            fft_high_ratios.append(ratio)

            # Spatial Blurring / Laplacian Variance (edge sharpness)
            laplacian = (
                -4 * frame +
                np.roll(frame, 1, axis=0) + np.roll(frame, -1, axis=0) +
                np.roll(frame, 1, axis=1) + np.roll(frame, -1, axis=1)
            )
            blurriness_scores.append(float(np.var(laplacian)))

        avg_fft_ratio = float(np.mean(fft_high_ratios))
        avg_blurriness = float(np.mean(blurriness_scores))

        # Evaluate FFT High Frequency Ratio
        if avg_fft_ratio > HEURISTIC_FFT_HIGH_FREQ_RATIO_THRESHOLD:
            signals.append(ForensicSignal(
                signal_name="high_frequency_energy_ratio",
                risk_level=RISK_LEVEL_MEDIUM,
                score=avg_fft_ratio,
                details=f"Elevated high-frequency energy ratio ({avg_fft_ratio:.4f}); potential spatial grid artifact.",
                evidence_strength="low",
                type="heuristic"
            ))
        else:
            signals.append(ForensicSignal(
                signal_name="high_frequency_energy_ratio",
                risk_level=RISK_LEVEL_LOW,
                score=avg_fft_ratio,
                details=f"Normal spatial frequency energy ratio ({avg_fft_ratio:.4f}).",
                evidence_strength="low",
                type="heuristic"
            ))

        # Evaluate Blurriness / Sharpness Signal
        if avg_blurriness < HEURISTIC_BLURRINESS_LAPLACIAN_THRESHOLD:
            signals.append(ForensicSignal(
                signal_name="spatial_blurriness_laplacian",
                risk_level=RISK_LEVEL_MEDIUM,
                score=avg_blurriness,
                details=f"Low spatial edge variance ({avg_blurriness:.2f}); indicates spatial smoothing or compression blur.",
                evidence_strength="low",
                type="heuristic"
            ))
        else:
            signals.append(ForensicSignal(
                signal_name="spatial_blurriness_laplacian",
                risk_level=RISK_LEVEL_LOW,
                score=avg_blurriness,
                details=f"Normal spatial edge variance ({avg_blurriness:.2f}).",
                evidence_strength="low",
                type="heuristic"
            ))

        # Signal C: Frame-to-Frame Temporal High-Frequency Variation
        if n_frames > 1:
            diffs = [np.abs(frames_data[i] - frames_data[i - 1]) for i in range(1, n_frames)]
            temporal_var = float(np.mean([np.var(d) for d in diffs]))
            if temporal_var > HEURISTIC_TEMPORAL_NOISE_VAR_THRESHOLD * 100.0:
                signals.append(ForensicSignal(
                    signal_name="temporal_noise_variation",
                    risk_level=RISK_LEVEL_MEDIUM,
                    score=temporal_var,
                    details=f"Elevated frame-to-frame temporal noise variance ({temporal_var:.2f}).",
                    evidence_strength="low",
                    type="heuristic"
                ))
            else:
                signals.append(ForensicSignal(
                    signal_name="temporal_noise_variation",
                    risk_level=RISK_LEVEL_LOW,
                    score=temporal_var,
                    details=f"Normal temporal noise variance ({temporal_var:.2f}).",
                    evidence_strength="low",
                    type="heuristic"
                ))

        return signals

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
