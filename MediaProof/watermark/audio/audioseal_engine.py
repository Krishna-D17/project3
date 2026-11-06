import math
import os
import shutil
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from media.ffmpeg import FFmpegExecutionError, get_ffmpeg_path, run_ffmpeg_command
from media.service import MediaService
from provenance.service import ProvenanceService
from provenance.repository import get_asset_by_watermark_token
from watermark.audio.base import BaseAudioWatermarkEngine, AudioDetectionResult, AudioEmbeddingResult
from watermark.audio.config import AUDIO_SAMPLE_RATE, DEVICE, DETECTION_CONFIDENCE_THRESHOLD

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import audioseal
    HAS_AUDIOSEAL = True
except ImportError:
    HAS_AUDIOSEAL = False


# Process-level model singleton cache
_MODEL_CACHE: Dict[str, Any] = {}


def hex_token_to_bits_tensor(token_hex: str) -> "torch.Tensor":
    """Converts a 4-character 16-bit hex token (e.g. 'A3F8') to a (1, 16) binary PyTorch tensor."""
    val = int(token_hex, 16)
    bits = [(val >> (15 - i)) & 1 for i in range(16)]
    return torch.tensor([bits], dtype=torch.int32)


def bits_array_to_hex_token(bits: List[int]) -> str:
    """Converts a 16-element list/tensor of binary bits to a 4-character upper-case hex string."""
    val = 0
    for b in bits[:16]:
        val = (val << 1) | (1 if b > 0 else 0)
    return f"{val:04X}"


def hamming_distance(bits1: List[int], bits2: List[int]) -> int:
    """Calculates Hamming distance between two 16-bit binary arrays."""
    return sum(b1 != b2 for b1, b2 in zip(bits1[:16], bits2[:16]))


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


class MetaAudioSealEngine(BaseAudioWatermarkEngine):
    """Meta / Facebook Research AudioSeal Neural Audio Watermarking Engine (CPU Inference)."""

    @property
    def engine_name(self) -> str:
        return "audioseal"

    @property
    def display_name(self) -> str:
        return "Meta AudioSeal Neural Audio Watermark Engine"

    def __init__(self, device: str = DEVICE, confidence_threshold: Optional[float] = None):
        self.device = device
        if confidence_threshold is not None:
            confidence_threshold = validate_detection_threshold(confidence_threshold, "confidence_threshold")
        self.confidence_threshold = confidence_threshold

    def _load_models(self):
        """Lazy-loads official Meta AudioSeal PyTorch generator and detector onto CPU."""
        if "generator" in _MODEL_CACHE and "detector" in _MODEL_CACHE:
            return _MODEL_CACHE["generator"], _MODEL_CACHE["detector"]

        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch is not installed in current environment. "
                "Install ML requirements via 'pip install -r requirements-ml.txt'."
            )

        if not HAS_AUDIOSEAL:
            raise RuntimeError(
                "Meta AudioSeal package is not installed in current environment. "
                "Install via '.venv\\Scripts\\python.exe -m pip install audioseal'."
            )

        try:
            from audioseal import AudioSeal
            generator = AudioSeal.load_generator("audioseal_wm_16bits").to(self.device)
            detector = AudioSeal.load_detector("audioseal_detector_16bits").to(self.device)
            generator.eval()
            detector.eval()

            _MODEL_CACHE["generator"] = generator
            _MODEL_CACHE["detector"] = detector
            return generator, detector

        except Exception as e:
            raise RuntimeError(f"Failed to load Meta AudioSeal pretrained models on CPU: {e}")

    def embed(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        asset_id: str,
        timeout: float = 60.0,
    ) -> AudioEmbeddingResult:
        start_time = time.time()
        generator, _ = self._load_models()

        in_p = Path(input_path).resolve()
        out_p = Path(output_path).resolve()

        if not in_p.exists() or not in_p.is_file():
            raise FileNotFoundError(f"Input audio file not found: {in_p}")

        ffmpeg_exe = get_ffmpeg_path()
        if not ffmpeg_exe or not ffmpeg_exe.exists():
            raise RuntimeError("FFmpeg executable is not available for AudioSeal watermarking.")

        probe_res = MediaService.probe_file(in_p)
        if not probe_res.has_audio:
            raise ValueError(f"File '{in_p.name}' does not contain an audio stream.")

        # Resolve or generate 16-bit watermark token mapped to SQLite asset record
        watermark_tok = ProvenanceService.ensure_watermark_token(asset_id)
        msg_tensor = hex_token_to_bits_tensor(watermark_tok).to(self.device)

        out_p.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="mp_aseal_embed_"))

        try:
            pcm_in_path = temp_dir / "input_16k.wav"
            pcm_out_path = temp_dir / "watermarked_16k.wav"

            # 1. Convert input audio to 16 kHz Mono 16-bit PCM WAV using FFmpeg
            cmd_conv = [
                str(ffmpeg_exe), "-y",
                "-i", str(in_p),
                "-ar", str(AUDIO_SAMPLE_RATE),
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(pcm_in_path)
            ]
            res_conv = run_ffmpeg_command(cmd_conv, timeout=timeout)
            if res_conv.returncode != 0 or not pcm_in_path.exists():
                raise FFmpegExecutionError(
                    message=f"Failed to extract audio stream to 16kHz PCM WAV: {res_conv.stderr}",
                    returncode=res_conv.returncode,
                    stdout=res_conv.stdout,
                    stderr=res_conv.stderr
                )

            # 2. Read PCM audio samples into PyTorch waveform tensor (1, 1, T)
            with wave.open(str(pcm_in_path), "rb") as wf:
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)
                int16_samples = np.frombuffer(raw_bytes, dtype=np.int16)
                audio_np = int16_samples.astype(np.float32) / 32767.0

            if len(audio_np) == 0:
                raise ValueError("Extracted audio stream contains 0 frames.")

            audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0).unsqueeze(0).to(self.device)

            # 3. AudioSeal PyTorch CPU embedding
            with torch.no_grad():
                watermark = generator.get_watermark(audio_tensor, sample_rate=AUDIO_SAMPLE_RATE, message=msg_tensor)
                watermarked_tensor = audio_tensor + watermark

            wm_np = watermarked_tensor.squeeze().cpu().numpy()
            int16_wm = np.clip(wm_np * 32767.0, -32768, 32767).astype(np.int16)

            # 4. Save watermarked 16kHz PCM WAV
            with wave.open(str(pcm_out_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(AUDIO_SAMPLE_RATE)
                wf.writeframes(int16_wm.tobytes())

            # 5. Export to destination path (WAV/MP3/AAC) using FFmpeg
            cmd_export = [
                str(ffmpeg_exe), "-y",
                "-i", str(pcm_out_path),
                str(out_p)
            ]
            res_exp = run_ffmpeg_command(cmd_export, timeout=timeout)
            if res_exp.returncode != 0 or not out_p.exists():
                raise FFmpegExecutionError(
                    message=f"Failed to export AudioSeal watermarked audio file: {res_exp.stderr}",
                    returncode=res_exp.returncode,
                    stdout=res_exp.stdout,
                    stderr=res_exp.stderr
                )

            proc_time = time.time() - start_time
            return AudioEmbeddingResult(
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
    ) -> AudioDetectionResult:
        start_time = time.time()
        if confidence_threshold is not None:
            effective_threshold = validate_detection_threshold(confidence_threshold, "confidence_threshold")
        elif self.confidence_threshold is not None:
            effective_threshold = validate_detection_threshold(self.confidence_threshold, "confidence_threshold")
        else:
            effective_threshold = validate_detection_threshold(DETECTION_CONFIDENCE_THRESHOLD, "default threshold")

        try:
            _, detector = self._load_models()
        except Exception as e:
            return AudioDetectionResult(
                engine=self.engine_name,
                watermark_detected=False,
                confidence=0.0,
                recovered_asset_id=None,
                provenance_record=None,
                processing_time_seconds=time.time() - start_time,
                message=f"Failed to load AudioSeal model: {str(e)}",
                details={"engine_display_name": self.display_name, "error": str(e)},
            )

        in_p = Path(input_path).resolve()
        if not in_p.exists() or not in_p.is_file():
            raise FileNotFoundError(f"Input audio file not found: {in_p}")

        ffmpeg_exe = get_ffmpeg_path()
        if not ffmpeg_exe or not ffmpeg_exe.exists():
            return AudioDetectionResult(
                engine=self.engine_name,
                watermark_detected=False,
                confidence=0.0,
                recovered_asset_id=None,
                provenance_record=None,
                processing_time_seconds=time.time() - start_time,
                message="FFmpeg binary unavailable for AudioSeal watermark detection.",
            )

        probe_res = MediaService.probe_file(in_p)
        if not probe_res.has_audio:
            return AudioDetectionResult(
                engine=self.engine_name,
                watermark_detected=False,
                confidence=0.0,
                recovered_asset_id=None,
                provenance_record=None,
                processing_time_seconds=time.time() - start_time,
                message="Media file contains no audio streams.",
            )

        temp_dir = Path(tempfile.mkdtemp(prefix="mp_aseal_detect_"))

        try:
            pcm_in_path = temp_dir / "detect_16k.wav"

            # 1. Convert input audio to 16 kHz Mono PCM WAV using FFmpeg
            cmd_conv = [
                str(ffmpeg_exe), "-y",
                "-i", str(in_p),
                "-ar", str(AUDIO_SAMPLE_RATE),
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(pcm_in_path)
            ]
            res_conv = run_ffmpeg_command(cmd_conv, timeout=timeout)
            if res_conv.returncode != 0 or not pcm_in_path.exists():
                return AudioDetectionResult(
                    engine=self.engine_name,
                    watermark_detected=False,
                    confidence=0.0,
                    recovered_asset_id=None,
                    provenance_record=None,
                    processing_time_seconds=time.time() - start_time,
                    message=f"FFmpeg audio extraction failed: {res_conv.stderr}",
                )

            # 2. Read PCM samples into 1D waveform tensor
            with wave.open(str(pcm_in_path), "rb") as wf:
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)
                int16_samples = np.frombuffer(raw_bytes, dtype=np.int16)
                audio_np = int16_samples.astype(np.float32) / 32767.0

            if len(audio_np) == 0:
                return AudioDetectionResult(
                    engine=self.engine_name,
                    watermark_detected=False,
                    confidence=0.0,
                    recovered_asset_id=None,
                    provenance_record=None,
                    processing_time_seconds=time.time() - start_time,
                    message="Extracted audio stream contains 0 frames.",
                )

            audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0).unsqueeze(0).to(self.device)

            # 3. AudioSeal PyTorch CPU detection
            try:
                with torch.no_grad():
                    result, message_logits = detector(audio_tensor, sample_rate=AUDIO_SAMPLE_RATE)
                presence_prob = float(torch.sigmoid(result).mean().item())
            except Exception as e:
                return AudioDetectionResult(
                    engine=self.engine_name,
                    watermark_detected=False,
                    confidence=0.0,
                    recovered_asset_id=None,
                    provenance_record=None,
                    processing_time_seconds=time.time() - start_time,
                    message=f"AudioSeal neural detection execution failed: {str(e)}",
                    details={"engine_display_name": self.display_name, "error": str(e)},
                )

            if not math.isfinite(presence_prob):
                return AudioDetectionResult(
                    engine=self.engine_name,
                    watermark_detected=False,
                    confidence=0.0,
                    recovered_asset_id=None,
                    provenance_record=None,
                    processing_time_seconds=time.time() - start_time,
                    message="AudioSeal neural detector produced invalid non-finite presence output.",
                    details={
                        "engine_display_name": self.display_name,
                        "error": "non_finite_presence",
                        "detection_threshold": effective_threshold,
                        "threshold_passed": False,
                    },
                )

            # Decode message logits shape: (1, 16) or (1, 16, Frames)
            if message_logits.dim() == 3:
                presence_mask = (torch.sigmoid(result) > 0.5)
                if presence_mask.any():
                    masked_logits = message_logits * presence_mask.float()
                    avg_logits = (masked_logits.sum(dim=-1) / presence_mask.float().sum(dim=-1).clamp(min=1.0)).squeeze(0)
                else:
                    avg_logits = message_logits.mean(dim=-1).squeeze(0)
            else:
                avg_logits = message_logits.squeeze(0)

            decoded_bits = [(1 if float(l) > 0.5 else 0) for l in avg_logits]
            raw_token = bits_array_to_hex_token(decoded_bits)

            # 1. Direct exact lookup in SQLite
            provenance_rec = get_asset_by_watermark_token(raw_token, allow_hamming=False, media_type="audio")
            if not provenance_rec:
                provenance_rec = get_asset_by_watermark_token(raw_token, allow_hamming=False)

            best_token = raw_token
            matching_method = "exact"
            matched_hamming_distance = 0

            # 2. Hamming distance fallback if exact match failed (distance <= 2)
            if not provenance_rec:
                rec_h2 = get_asset_by_watermark_token(raw_token, allow_hamming=True, media_type="audio")
                if not rec_h2:
                    rec_h2 = get_asset_by_watermark_token(raw_token, allow_hamming=True)

                if rec_h2:
                    provenance_rec = rec_h2
                    best_token = rec_h2.watermark_token
                    matching_method = "hamming_fallback"
                    try:
                        ref_bits = hex_token_to_bits_tensor(best_token).squeeze(0).tolist()
                        matched_hamming_distance = hamming_distance(decoded_bits, ref_bits)
                    except Exception:
                        matched_hamming_distance = None

            # 3. Minimum Hamming distance lookup across registered audio SQLite tokens (distance <= 4)
            if not provenance_rec:
                from provenance.repository import get_db_connection
                conn = get_db_connection()
                try:
                    cursor = conn.execute("SELECT watermark_token FROM assets WHERE media_type = 'audio' AND watermark_token IS NOT NULL")
                    registered_tokens = [row["watermark_token"] for row in cursor.fetchall() if len(row["watermark_token"]) == 4]
                finally:
                    conn.close()

                min_hdist = 16
                best_match_tok = None
                for tok_str in registered_tokens:
                    try:
                        ref_bits = hex_token_to_bits_tensor(tok_str).squeeze(0).tolist()
                        hdist = hamming_distance(decoded_bits, ref_bits)
                        if hdist < min_hdist:
                            min_hdist = hdist
                            best_match_tok = tok_str
                    except Exception:
                        pass

                if min_hdist <= 4 and best_match_tok:
                    rec_fallback = ProvenanceService.find_asset_by_watermark_token(best_match_tok)
                    if rec_fallback:
                        provenance_rec = rec_fallback
                        best_token = best_match_tok
                        matching_method = "hamming_fallback"
                        matched_hamming_distance = min_hdist

            confidence = round(presence_prob, 4)

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
                        f"AudioSeal neural presence confidence ({confidence:.4f}) is below detection threshold ({effective_threshold:.2f}) "
                        "and no registered watermark token match found."
                    )
                return AudioDetectionResult(
                    engine=self.engine_name,
                    watermark_detected=False,
                    confidence=confidence,
                    recovered_asset_id=None,
                    provenance_record=None,
                    processing_time_seconds=time.time() - start_time,
                    message=msg,
                    details={
                        "raw_decoded_token": raw_token,
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
                return AudioDetectionResult(
                    engine=self.engine_name,
                    watermark_detected=False,
                    confidence=confidence,
                    recovered_asset_id=None,
                    provenance_record=None,
                    processing_time_seconds=time.time() - start_time,
                    message="AudioSeal neural presence detected, but no registered MediaProof watermark token match found.",
                    details={
                        "raw_decoded_token": raw_token,
                        "engine_display_name": self.display_name,
                        "detection_threshold": effective_threshold,
                        "threshold_passed": True,
                    },
                )

            recovered_asset_id = provenance_rec.asset_id

            if matching_method == "hamming_fallback":
                message = (
                    f"AudioSeal watermark payload detected using Hamming distance fallback match "
                    f"(distance={matched_hamming_distance})."
                )
            else:
                message = "AudioSeal watermark payload successfully detected and decoded."

            return AudioDetectionResult(
                engine=self.engine_name,
                watermark_detected=True,
                confidence=confidence,
                recovered_asset_id=recovered_asset_id,
                provenance_record=provenance_rec.to_dict(),
                processing_time_seconds=time.time() - start_time,
                message=message,
                details={
                    "watermark_token": best_token,
                    "raw_decoded_token": raw_token,
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
