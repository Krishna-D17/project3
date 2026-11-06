import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Union

from media.ffmpeg import FFmpegExecutionError, get_ffmpeg_path, run_ffmpeg_command
from media.service import MediaService
from provenance.service import ProvenanceService
from watermark.video.base import BaseVideoWatermarkEngine, DetectionResult, EmbeddingResult
from watermark.video.config import DETECTION_CONFIDENCE_THRESHOLD, DEVICE, PAYLOAD_BIT_LENGTH
from watermark.video.payload import decode_asset_id_payload, encode_asset_id_payload


class CustomVideoWatermarkEngine(BaseVideoWatermarkEngine):
    """MediaProof Custom Spatial Video Watermark Engine.
    
    A fast, deterministic CPU baseline engine using vertical spatial pair modulation on the Green channel.
    """

    @property
    def engine_name(self) -> str:
        return "custom"

    @property
    def display_name(self) -> str:
        return "MediaProof Custom Spatial Video Watermark"

    def __init__(self, device: str = DEVICE):
        self.device = device

    def embed_frame_pixels(self, frame_bytes: bytearray, width: int, height: int, payload_bits: List[int]) -> bytearray:
        """Embeds 256-bit payload using vertical spatial pair modulation on Green/Luma channel."""
        output = bytearray(frame_bytes)
        num_pixels = width * height
        num_bits = len(payload_bits)
        if num_pixels == 0 or num_bits == 0 or height < 2:
            return output

        usable_rows = (height // 2) * 2
        usable_pixels = width * usable_rows
        block_size = max(2, (usable_pixels // 2) // num_bits)
        delta = 16

        for i, bit in enumerate(payload_bits):
            block_start = i * block_size
            block_end = min(usable_pixels // 2, (i + 1) * block_size)

            for idx in range(block_start, block_end):
                col = idx % width
                row = (idx // width) * 2
                if row + 1 >= height:
                    continue

                even_offset = (row * width + col) * 3 + 1  # Green channel
                odd_offset = ((row + 1) * width + col) * 3 + 1

                if odd_offset < len(output):
                    val_even = output[even_offset]
                    val_odd = output[odd_offset]
                    if bit == 1:
                        output[even_offset] = min(255, val_even + delta)
                        output[odd_offset] = max(0, val_odd - delta)
                    else:
                        output[even_offset] = max(0, val_even - delta)
                        output[odd_offset] = min(255, val_odd + delta)

        return output

    def detect_frame_pixels(self, frame_bytes: bytes, width: int, height: int) -> List[float]:
        """Detects 256-bit payload predictions using vertical spatial pair analysis on Green/Luma channel."""
        num_pixels = width * height
        num_bits = PAYLOAD_BIT_LENGTH
        if num_pixels == 0 or height < 2:
            return [0.0] * num_bits

        usable_rows = (height // 2) * 2
        usable_pixels = width * usable_rows
        block_size = max(2, (usable_pixels // 2) // num_bits)
        predictions = []

        for i in range(num_bits):
            block_start = i * block_size
            block_end = min(usable_pixels // 2, (i + 1) * block_size)

            diff_sum = 0
            count = 0

            for idx in range(block_start, block_end):
                col = idx % width
                row = (idx // width) * 2
                if row + 1 >= height:
                    continue

                even_offset = (row * width + col) * 3 + 1
                odd_offset = ((row + 1) * width + col) * 3 + 1

                if odd_offset < len(frame_bytes):
                    val_even = int(frame_bytes[even_offset])
                    val_odd = int(frame_bytes[odd_offset])
                    diff_sum += (val_even - val_odd)
                    count += 1

            if count > 0:
                avg_diff = diff_sum / count
                predictions.append(1.0 if avg_diff > 0 else 0.0)
            else:
                predictions.append(0.0)

        return predictions

    def embed(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        asset_id: str,
        timeout: float = 60.0,
    ) -> EmbeddingResult:
        start_time = time.time()
        in_p = Path(input_path).resolve()
        out_p = Path(output_path).resolve()

        if not in_p.exists() or not in_p.is_file():
            raise FileNotFoundError(f"Input video file not found: {in_p}")

        ffmpeg_exe = get_ffmpeg_path()
        if not ffmpeg_exe or not ffmpeg_exe.exists():
            raise RuntimeError("FFmpeg executable is not available for video watermarking.")

        probe_res = MediaService.probe_file(in_p)
        if not probe_res.has_video or not probe_res.video_streams:
            raise ValueError(f"File '{in_p.name}' does not contain a video stream.")

        v_stream = probe_res.video_streams[0]
        width, height = v_stream.width, v_stream.height
        fps = v_stream.fps if v_stream.fps > 0 else 25.0

        payload_bits = encode_asset_id_payload(asset_id, sha256_hex=probe_res.raw_format_tags.get("sha256"))

        out_p.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="mp_custom_embed_"))

        try:
            raw_in_pipe = temp_dir / "raw_in.rgb"
            raw_out_pipe = temp_dir / "raw_out.rgb"

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

            frame_size = width * height * 3

            with open(raw_in_pipe, "rb") as fin, open(raw_out_pipe, "wb") as fout:
                while True:
                    frame_data = fin.read(frame_size)
                    if not frame_data:
                        break
                    if len(frame_data) == frame_size:
                        wm_frame = self.embed_frame_pixels(bytearray(frame_data), width, height, payload_bits)
                        fout.write(wm_frame)
                    else:
                        fout.write(frame_data)

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
                    "-pix_fmt", "yuv420p",
                    "-c:a", "copy",
                    "-map", "0:v:0",
                    "-map", "1:a:0?",
                ])
            else:
                cmd_reencode.extend([
                    "-c:v", "libx264",
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
                details={"engine_display_name": self.display_name},
            )

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def detect(
        self,
        input_path: Union[str, Path],
        timeout: float = 30.0,
    ) -> DetectionResult:
        start_time = time.time()
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
                message="FFmpeg binary unavailable for watermark detection.",
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

        temp_dir = Path(tempfile.mkdtemp(prefix="mp_custom_detect_"))

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

            accumulated_bits = [0.0] * PAYLOAD_BIT_LENGTH
            frame_count = 0

            with open(raw_in_pipe, "rb") as fin:
                while True:
                    frame_data = fin.read(frame_size)
                    if not frame_data or len(frame_data) < frame_size:
                        break
                    preds = self.detect_frame_pixels(frame_data, width, height)
                    for idx, val in enumerate(preds):
                        accumulated_bits[idx] += val
                    frame_count += 1

            if frame_count == 0:
                return DetectionResult(
                    engine=self.engine_name,
                    watermark_detected=False,
                    confidence=0.0,
                    recovered_asset_id=None,
                    provenance_record=None,
                    processing_time_seconds=time.time() - start_time,
                    message="No valid frames found in video.",
                )

            avg_predictions = [val / frame_count for val in accumulated_bits]
            binary_bits = [1 if val >= 0.5 else 0 for val in avg_predictions]

            decoded = decode_asset_id_payload(binary_bits)

            if not decoded.get("valid") or not decoded.get("asset_id"):
                return DetectionResult(
                    engine=self.engine_name,
                    watermark_detected=False,
                    confidence=0.0,
                    recovered_asset_id=None,
                    provenance_record=None,
                    processing_time_seconds=time.time() - start_time,
                    message="No valid MediaProof watermark payload found.",
                )

            recovered_asset_id = str(decoded["asset_id"])

            confidence_accum = sum(abs(val - 0.5) * 2.0 for val in avg_predictions) / PAYLOAD_BIT_LENGTH
            confidence = round(confidence_accum, 4)

            provenance_rec = ProvenanceService.get_asset(recovered_asset_id)
            prov_dict = provenance_rec.to_dict() if provenance_rec else None

            detected = confidence >= DETECTION_CONFIDENCE_THRESHOLD or prov_dict is not None

            return DetectionResult(
                engine=self.engine_name,
                watermark_detected=detected,
                confidence=confidence,
                recovered_asset_id=recovered_asset_id,
                provenance_record=prov_dict,
                processing_time_seconds=time.time() - start_time,
                message="Watermark payload successfully detected and decoded." if detected else "Weak watermark signal.",
                details={"engine_display_name": self.display_name},
            )

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
