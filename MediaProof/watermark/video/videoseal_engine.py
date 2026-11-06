import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from media.ffmpeg import FFmpegExecutionError, get_ffmpeg_path, run_ffmpeg_command
from media.service import MediaService
from provenance.service import ProvenanceService
from provenance.repository import get_asset_by_watermark_token
from watermark.video.base import BaseVideoWatermarkEngine, DetectionResult, EmbeddingResult
from watermark.video.config import VIDEOSEAL_DETECTION_CONFIDENCE_THRESHOLD, DEVICE, MODEL_CHECKPOINT_PATH, PAYLOAD_BIT_LENGTH
from watermark.video.payload import decode_watermark_token_payload, encode_watermark_token_payload

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import videoseal
    HAS_VIDEOSEAL_PKG = True
except ImportError:
    HAS_VIDEOSEAL_PKG = False


def validate_detection_threshold(threshold: Any, param_name: str = "confidence_threshold") -> float:
    """Validates that a detection threshold is a finite numeric float/int in [0.0, 1.0].

    Rejects booleans, strings, non-numeric types, NaNs, and infinities with ValueError.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(
            f"Invalid {param_name}: {threshold!r}. Threshold must be a numeric float or int in [0.0, 1.0], not {type(threshold).__name__}."
        )
    val = float(threshold)
    if not math.isfinite(val) or val < 0.0 or val > 1.0:
        raise ValueError(
            f"Invalid {param_name}: {threshold!r}. Threshold must be a finite number between 0.0 and 1.0 inclusive."
        )
    return val


class MetaVideoSealEngine(BaseVideoWatermarkEngine):
    """Meta / Facebook Research VideoSeal Neural Watermarking Engine (CPU Inference)."""

    @property
    def engine_name(self) -> str:
        return "videoseal"

    @property
    def display_name(self) -> str:
        return "Meta VideoSeal Neural Watermark Engine"

    def __init__(self, device: str = DEVICE, confidence_threshold: Optional[float] = None):
        self.device = device
        if confidence_threshold is not None:
            confidence_threshold = validate_detection_threshold(confidence_threshold, "confidence_threshold")
        self.confidence_threshold = confidence_threshold
        self.model = None

    def _load_model(self):
        """Lazy-loads official Meta VideoSeal PyTorch model onto CPU."""
        if self.model is not None:
            return self.model

        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch is not installed in current environment. "
                "Install ML requirements via 'pip install -r requirements-ml.txt'."
            )

        if HAS_VIDEOSEAL_PKG:
            try:
                self.model = videoseal.load("videoseal").to(self.device)
                self.model.step_size = 1  # Watermark all frames for maximum neural bit stability
                self.model.eval()
                return self.model
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Meta VideoSeal model from videoseal package: {e}")

        if MODEL_CHECKPOINT_PATH.exists() and MODEL_CHECKPOINT_PATH.stat().st_size > 100000:
            try:
                checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=self.device)
                self.model = checkpoint
                if hasattr(self.model, "step_size"):
                    self.model.step_size = 1
                return self.model
            except Exception as e:
                raise RuntimeError(f"Failed to load VideoSeal checkpoint from '{MODEL_CHECKPOINT_PATH}': {e}")

        raise RuntimeError(
            f"Meta VideoSeal model checkpoint missing at '{MODEL_CHECKPOINT_PATH}'. "
            "Run '.venv\\Scripts\\python.exe scripts/download_videoseal_models.py' to download official weights."
        )

    def embed(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        asset_id: str,
        timeout: float = 60.0,
    ) -> EmbeddingResult:
        start_time = time.time()
        model = self._load_model()

        in_p = Path(input_path).resolve()
        out_p = Path(output_path).resolve()

        if not in_p.exists() or not in_p.is_file():
            raise FileNotFoundError(f"Input video file not found: {in_p}")

        ffmpeg_exe = get_ffmpeg_path()
        if not ffmpeg_exe or not ffmpeg_exe.exists():
            raise RuntimeError("FFmpeg executable is not available for VideoSeal watermarking.")

        probe_res = MediaService.probe_file(in_p)
        if not probe_res.has_video or not probe_res.video_streams:
            raise ValueError(f"File '{in_p.name}' does not contain a video stream.")

        v_stream = probe_res.video_streams[0]
        width, height = v_stream.width, v_stream.height
        fps = v_stream.fps if v_stream.fps > 0 else 25.0

        # Resolve or generate watermark token mapped to SQLite asset record
        watermark_tok = ProvenanceService.ensure_watermark_token(asset_id)
        payload_bits = encode_watermark_token_payload(watermark_tok)

        out_p.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="mp_vseal_embed_"))

        try:
            raw_in_pipe = temp_dir / "raw_in.rgb"
            raw_out_pipe = temp_dir / "raw_out.rgb"

            # 1. Extract RGB24 video frames using FFmpeg
            cmd_extract = [
                str(ffmpeg_exe), "-y",
                "-i", str(in_p),
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                str(raw_in_pipe)
            ]
            res_ext = run_ffmpeg_command(cmd_extract, timeout=timeout)
            if res_ext.returncode != 0 or not raw_in_pipe.exists():
                raise FFmpegExecutionError(
                    message=f"Failed to extract raw video frames: {res_ext.stderr}",
                    returncode=res_ext.returncode,
                    stdout=res_ext.stdout,
                    stderr=res_ext.stderr
                )

            # 2. Read frames into a batched PyTorch Video Tensor (F, 3, H, W)
            frame_size = width * height * 3
            frame_list = []

            with open(raw_in_pipe, "rb") as fin:
                while True:
                    frame_data = fin.read(frame_size)
                    if not frame_data or len(frame_data) < frame_size:
                        break
                    img_np = torch.frombuffer(bytearray(frame_data), dtype=torch.uint8).reshape(height, width, 3)
                    frame_list.append(img_np.permute(2, 0, 1).float() / 255.0)

            if not frame_list:
                raise ValueError("No video frames extracted from input media.")

            video_tensor = torch.stack(frame_list, dim=0)  # Shape: (F, 3, H, W)
            msg_tensor = torch.tensor([payload_bits], dtype=torch.int64, device=self.device)

            with torch.no_grad():
                outputs = model.embed(video_tensor.to(self.device), msg_tensor, is_video=True)
                wm_video_tensor = outputs["imgs_w"]

            with open(raw_out_pipe, "wb") as fout:
                for i in range(wm_video_tensor.shape[0]):
                    wm_frame_np = (wm_video_tensor[i].permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
                    fout.write(wm_frame_np.tobytes())

            # 3. Re-encode video and multiplex original audio track using FFmpeg (CRF 14 for feature preservation)
            cmd_reencode = [
                str(ffmpeg_exe), "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", str(raw_out_pipe),
            ]

            if probe_res.has_audio:
                cmd_reencode.extend([
                    "-i", str(in_p),
                    "-c:v", "libx264",
                    "-crf", "14",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "copy",
                    "-map", "0:v:0",
                    "-map", "1:a:0?",
                ])
            else:
                cmd_reencode.extend([
                    "-c:v", "libx264",
                    "-crf", "14",
                    "-pix_fmt", "yuv420p",
                ])

            cmd_reencode.append(str(out_p))

            res_enc = run_ffmpeg_command(cmd_reencode, timeout=timeout)
            if res_enc.returncode != 0 or not out_p.exists():
                raise FFmpegExecutionError(
                    message=f"Failed to re-encode watermarked video: {res_enc.stderr}",
                    returncode=res_enc.returncode,
                    stdout=res_enc.stdout,
                    stderr=res_enc.stderr
                )

            proc_time = time.time() - start_time
            return EmbeddingResult(
                engine=self.engine_name,
                output_path=str(out_p),
                asset_id=asset_id,
                size_bytes=out_p.stat().st_size,
                has_watermark=True,
                processing_time_seconds=proc_time,
                details={
                    "engine_display_name": self.display_name,
                    "watermark_token": watermark_tok,
                },
            )

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def detect(
        self,
        input_path: Union[str, Path],
        timeout: float = 30.0,
        confidence_threshold: Optional[float] = None,
    ) -> DetectionResult:
        start_time = time.time()
        if confidence_threshold is not None:
            effective_threshold = validate_detection_threshold(confidence_threshold, "confidence_threshold")
        elif self.confidence_threshold is not None:
            effective_threshold = validate_detection_threshold(self.confidence_threshold, "confidence_threshold")
        else:
            effective_threshold = validate_detection_threshold(VIDEOSEAL_DETECTION_CONFIDENCE_THRESHOLD, "default threshold")

        try:
            model = self._load_model()
        except Exception as e:
            return DetectionResult(
                engine=self.engine_name,
                watermark_detected=False,
                confidence=0.0,
                recovered_asset_id=None,
                provenance_record=None,
                processing_time_seconds=time.time() - start_time,
                message=f"Failed to load VideoSeal model: {str(e)}",
                details={"engine_display_name": self.display_name, "error": str(e)},
            )

        in_p = Path(input_path).resolve()
        if not in_p.exists() or not in_p.is_file():
            raise FileNotFoundError(f"Input video file not found: {in_p}")

        ffmpeg_exe = get_ffmpeg_path()
        if not ffmpeg_exe or not ffmpeg_exe.exists():
            return DetectionResult(
                engine=self.engine_name,
                watermark_detected=False,
                confidence=0.0,
                recovered_asset_id=None,
                provenance_record=None,
                processing_time_seconds=time.time() - start_time,
                message="FFmpeg binary unavailable for VideoSeal watermark detection.",
            )

        probe_res = MediaService.probe_file(in_p)
        if not probe_res.has_video or not probe_res.video_streams:
            return DetectionResult(
                engine=self.engine_name,
                watermark_detected=False,
                confidence=0.0,
                recovered_asset_id=None,
                provenance_record=None,
                processing_time_seconds=time.time() - start_time,
                message="Media file contains no video streams.",
            )

        v_stream = probe_res.video_streams[0]
        width, height = v_stream.width, v_stream.height

        temp_dir = Path(tempfile.mkdtemp(prefix="mp_vseal_detect_"))

        try:
            raw_in_pipe = temp_dir / "raw_detect.rgb"

            cmd_extract = [
                str(ffmpeg_exe), "-y",
                "-i", str(in_p),
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                str(raw_in_pipe)
            ]
            res_ext = run_ffmpeg_command(cmd_extract, timeout=timeout)
            if res_ext.returncode != 0 or not raw_in_pipe.exists():
                return DetectionResult(
                    engine=self.engine_name,
                    watermark_detected=False,
                    confidence=0.0,
                    recovered_asset_id=None,
                    provenance_record=None,
                    processing_time_seconds=time.time() - start_time,
                    message=f"FFmpeg frame extraction failed: {res_ext.stderr}",
                )

            frame_size = width * height * 3
            frame_list = []

            with open(raw_in_pipe, "rb") as fin:
                while True:
                    frame_data = fin.read(frame_size)
                    if not frame_data or len(frame_data) < frame_size:
                        break
                    img_np = torch.frombuffer(bytearray(frame_data), dtype=torch.uint8).reshape(height, width, 3)
                    frame_list.append(img_np.permute(2, 0, 1).float() / 255.0)

            if not frame_list:
                return DetectionResult(
                    engine=self.engine_name,
                    watermark_detected=False,
                    confidence=0.0,
                    recovered_asset_id=None,
                    provenance_record=None,
                    processing_time_seconds=time.time() - start_time,
                    message="No valid frames found in video.",
                )

            video_tensor = torch.stack(frame_list, dim=0)

            try:
                with torch.no_grad():
                    det_out = model.detect(video_tensor.to(self.device), is_video=True)
                    preds = det_out["preds"]  # Shape: (F, 257)

                presence_logits = preds[:, 0]
                bit_logits = preds[:, 1:]

                avg_presence = float(torch.sigmoid(presence_logits).mean().item())
                avg_bit_logits = bit_logits.mean(dim=0).cpu().tolist()
            except Exception as e:
                return DetectionResult(
                    engine=self.engine_name,
                    watermark_detected=False,
                    confidence=0.0,
                    recovered_asset_id=None,
                    provenance_record=None,
                    processing_time_seconds=time.time() - start_time,
                    message=f"VideoSeal neural detection execution failed: {str(e)}",
                    details={"engine_display_name": self.display_name, "error": str(e)},
                )

            if not math.isfinite(avg_presence):
                return DetectionResult(
                    engine=self.engine_name,
                    watermark_detected=False,
                    confidence=0.0,
                    recovered_asset_id=None,
                    provenance_record=None,
                    processing_time_seconds=time.time() - start_time,
                    message="VideoSeal neural detector produced invalid non-finite presence output.",
                    details={
                        "engine_display_name": self.display_name,
                        "error": "non_finite_presence",
                        "detection_threshold": effective_threshold,
                        "threshold_passed": False,
                    },
                )

            # 1. Attempt 16-bit token decoding (16x repetition)
            recovered_token = decode_watermark_token_payload(avg_bit_logits, token_bits=16)

            # Exact match lookup
            provenance_rec = get_asset_by_watermark_token(recovered_token, allow_hamming=False, media_type="video")
            if not provenance_rec:
                provenance_rec = get_asset_by_watermark_token(recovered_token, allow_hamming=False)

            best_token = recovered_token
            matching_method = "exact"
            matched_hamming_distance = 0

            # Hamming fallback lookup (16-bit)
            if not provenance_rec:
                rec_h = get_asset_by_watermark_token(recovered_token, allow_hamming=True, media_type="video")
                if not rec_h:
                    rec_h = get_asset_by_watermark_token(recovered_token, allow_hamming=True)

                if rec_h:
                    provenance_rec = rec_h
                    best_token = rec_h.watermark_token
                    matching_method = "hamming_fallback"
                    try:
                        v1, v2 = int(recovered_token, 16), int(best_token, 16)
                        matched_hamming_distance = bin(v1 ^ v2).count("1")
                    except Exception:
                        matched_hamming_distance = None

            # 2. Attempt 32-bit token decoding fallback (8x repetition)
            if not provenance_rec:
                try:
                    tok32 = decode_watermark_token_payload(avg_bit_logits, token_bits=32)
                    rec_32 = ProvenanceService.find_asset_by_watermark_token(tok32)
                    if rec_32:
                        provenance_rec = rec_32
                        best_token = tok32
                        matching_method = "exact_32bit"
                        matched_hamming_distance = 0
                except Exception:
                    pass

            confidence = round(avg_presence, 4)

            # Neural presence gating: A database token match must NOT establish detection if presence < threshold
            if confidence < effective_threshold:
                cand_asset_id = provenance_rec.asset_id if provenance_rec else None
                if provenance_rec:
                    msg = (
                        f"Candidate token '{best_token}' matched asset '{cand_asset_id}', "
                        f"but neural presence confidence ({confidence:.4f}) is below detection threshold ({effective_threshold:.2f})."
                    )
                else:
                    msg = (
                        f"VideoSeal neural presence confidence ({confidence:.4f}) is below detection threshold ({effective_threshold:.2f}) "
                        "and no registered watermark token match found."
                    )
                return DetectionResult(
                    engine=self.engine_name,
                    watermark_detected=False,
                    confidence=confidence,
                    recovered_asset_id=None,
                    provenance_record=None,
                    processing_time_seconds=time.time() - start_time,
                    message=msg,
                    details={
                        "recovered_token": recovered_token,
                        "candidate_token": best_token if provenance_rec else None,
                        "candidate_asset_id": cand_asset_id,
                        "matching_method": matching_method if provenance_rec else None,
                        "hamming_distance": matched_hamming_distance if provenance_rec else None,
                        "engine_display_name": self.display_name,
                        "presence_confidence": confidence,
                        "detection_threshold": effective_threshold,
                        "threshold_passed": False,
                        "rejected_reason": "presence_below_threshold",
                    },
                )

            if not provenance_rec:
                return DetectionResult(
                    engine=self.engine_name,
                    watermark_detected=False,
                    confidence=confidence,
                    recovered_asset_id=None,
                    provenance_record=None,
                    processing_time_seconds=time.time() - start_time,
                    message="VideoSeal neural presence detected, but no registered MediaProof watermark token match found.",
                    details={
                        "recovered_token": recovered_token,
                        "engine_display_name": self.display_name,
                        "detection_threshold": effective_threshold,
                        "threshold_passed": True,
                    },
                )

            recovered_asset_id = provenance_rec.asset_id

            if matching_method == "hamming_fallback":
                message = (
                    f"VideoSeal watermark payload detected using Hamming distance fallback match "
                    f"(distance={matched_hamming_distance})."
                )
            else:
                message = "VideoSeal watermark payload successfully detected and decoded."

            return DetectionResult(
                engine=self.engine_name,
                watermark_detected=True,
                confidence=confidence,
                recovered_asset_id=recovered_asset_id,
                provenance_record=provenance_rec.to_dict(),
                processing_time_seconds=time.time() - start_time,
                message=message,
                details={
                    "watermark_token": best_token,
                    "recovered_token": recovered_token,
                    "engine_display_name": self.display_name,
                    "matching_method": matching_method,
                    "hamming_distance": matched_hamming_distance,
                    "detection_threshold": effective_threshold,
                    "threshold_passed": True,
                },
            )

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
